import argparse
import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import fake_hm
from common import char_count, read_jsonl, strict_loads, write_jsonl
from contracts import check_prediction, check_reference
from gen_truth import check_truth, generate, reference_id, scan_images, task_config, validate_reference_for_input
from evaluate import evaluate
from gen_html import aggregate, main as report_main, validate_result
from metrics import calculate, recommendation_score


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        # Valid one-pixel PNG; all semantic responses are synthetic, not image observations.
        (self.images/'menu.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jYuoAAAAASUVORK5CYII='))
        self.api = self.root/'api.json'
        self.api.write_text(json.dumps({'account':'test-only','sdk_module':'fake_hm','reference_model':'mock-reference',
            'reference_version':'fixture-v1','judge_model':'mock-judge','judge_version':'fixture-v1',
            'max_retries':2,'retry_wait':0}),encoding='utf-8')
        config = task_config()
        config['context']={'locale':'FR','acceptable_languages':['fr']}
        self.config = self.root/'tasks.json'
        self.config.write_text(json.dumps(config,ensure_ascii=False),encoding='utf-8')
        self.truth_path = self.root/'truth.jsonl'
        self.pred_path = self.root/'predictions.jsonl'
        self.eval_path = self.root/'evaluations.jsonl'
        fake_hm.reset()

    def tearDown(self):
        self.tmp.cleanup()

    def truth_args(self, **overrides):
        values=dict(images=self.images,output=self.truth_path,api_config=self.api,task_config=self.config,
                    manifest=None,export_inputs=self.root/'inputs.jsonl',prepare_only=False,resume=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def eval_args(self, **overrides):
        values=dict(truth=self.truth_path,predictions=self.pred_path,output=self.eval_path,api_config=self.api,
                    tasks=['summary','interpretation','recommendation','phrase'],model_id=None,run_id=None,
                    prepare_only=False,resume=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def build(self):
        result=generate(self.truth_args())
        self.assertEqual(result['errors'],0)
        t=read_jsonl(self.truth_path)[0]
        predictions=[]
        for task in t['tasks']:
            predictions.append({'schema_version':'fg-eval/1.1','image_name':'menu.png','image_sha256':t['image_sha256'],
                'task':task,'model_id':'target','run_id':'r1','prompt':t['tasks'][task]['prompt'], 'context':t['context'],
                'status':'ok','output':fake_hm.OUTPUTS[task],'error':None,
                'generation':{'elapsed_seconds':1.0,'input_tokens':None,'output_tokens':None,
                              'max_output_tokens':1024,'finish_reason':None}})
        write_jsonl(self.pred_path,predictions)
        return t,predictions

    def test_complete_sdk_pipeline_and_self_contained_report(self):
        t,p=self.build()
        self.assertEqual(len(fake_hm.CALLS),5)
        result=evaluate(self.eval_args())
        self.assertEqual(result['pending_reference_or_judge'],0)
        rows=read_jsonl(self.eval_path)
        self.assertEqual([r['score'] for r in rows],[1,1,1,1])
        for r in rows:validate_result(r)
        out=self.root/'report.html'
        report_main([str(self.eval_path),'--output',str(out),'--summary-output',str(self.root/'summary.json'),'--title','<script>test</script>'])
        html=out.read_text(encoding='utf-8')
        self.assertIn('data:image/png;base64,',html)
        self.assertNotIn('<script>test</script>',html)
        self.assertIn('&lt;script&gt;test&lt;/script&gt;',html)
        self.assertIn(p[0]['output'],html)
        self.assertFalse(t['human_verified'])
        n=len(fake_hm.CALLS)
        generate(self.truth_args(resume=True))
        evaluate(self.eval_args(resume=True))
        self.assertEqual(len(fake_hm.CALLS),n)

    def test_relative_length_precision_and_coverage(self):
        t,_=self.build()
        j=fake_hm.judge('summary')
        result=calculate(t,'summary',j,fake_hm.OUTPUTS['summary']*2)
        self.assertAlmostEqual(result['score'],0.5)
        j['claims'][1].update(label='supported_irrelevant',covered_ids=[])
        result=calculate(t,'summary',j,fake_hm.OUTPUTS['summary'])
        self.assertAlmostEqual(result['score'],0.5)

    def test_interpretation_partial_fields_and_extra_penalty(self):
        t,_=self.build()
        j=fake_hm.judge('interpretation')
        j['items'][1].update(status='missing',quote='',evidence_ids=[])
        j['extra_claims']=[{'id':'x1','text':'无证据','quote':'这是菜单','label':'unsupported',
            'related_item_id':None,'evidence_ids':[],'reason':'fixture'}]
        result=calculate(t,'interpretation',j,fake_hm.OUTPUTS['interpretation'])
        self.assertAlmostEqual(result['score'],0.5)
        self.assertIn('i2',result['details']['critical_failures'])

    def test_combination_partial_and_phrase_independence(self):
        t,_=self.build()
        j=fake_hm.judge('recommendation')
        j['selections'][0]['quantity']=2
        # Valid input quote but intentionally wrong quantities/amounts are never silently repaired.
        result=calculate(t,'recommendation',j,fake_hm.OUTPUTS['recommendation'])
        self.assertEqual(result['details']['computed_total_minor'],3200)
        self.assertIn('budget',result['details']['critical_failures'])
        self.assertAlmostEqual(result['score'],4/6)
        self.assertEqual(calculate(t,'phrase',fake_hm.judge('phrase'),fake_hm.OUTPUTS['phrase'])['score'],1)
        j['selections'][0]['quantity']=None
        self.assertNotEqual(recommendation_score(t,j)['score'],1)
        j=fake_hm.judge('recommendation');j['amount_claims'][0]['value_minor']=None
        result=calculate(t,'recommendation',j,fake_hm.OUTPUTS['recommendation'])
        self.assertEqual(next(c['status'] for c in result['details']['constraints'] if c['id']=='arithmetic_consistency'),'failed')

    def test_no_guessed_language_or_allergen_safety(self):
        ref=fake_hm.reference()
        with self.assertRaises(ValueError):check_reference(ref,{'locale':None,'acceptable_languages':[]})
        t,_=self.build()
        t=copy.deepcopy(t)
        t['reference']['recommendation']['menu'][0]['ingredients_complete']=False
        result=recommendation_score(t,fake_hm.judge('recommendation'))
        self.assertEqual(next(c['status'] for c in result['details']['constraints'] if c['id']=='allergen:shrimp'),'unknown')

    def test_missing_predictions_zero_but_judge_outage_not_zero(self):
        _,p=self.build()
        write_jsonl(self.pred_path,p[:1])
        fake_hm.FAIL_JUDGE=True
        evaluate(self.eval_args())
        rows=read_jsonl(self.eval_path)
        self.assertIsNone(rows[0]['score'])
        self.assertEqual([r['score'] for r in rows[1:]],[0,0,0])
        a=aggregate(rows)
        self.assertTrue(a['mean_is_partial'])
        self.assertEqual(a['judge_errors'],1)

    def test_prompt_mismatch_and_stale_resume_rejected(self):
        _,p=self.build()
        p[0]['prompt']['user']='different prompt'
        write_jsonl(self.pred_path,p)
        with self.assertRaisesRegex(ValueError,'prompt/context'):evaluate(self.eval_args())
        config=json.loads(self.config.read_text(encoding='utf-8'))
        config['tasks']['summary']['prompt']['user']='changed'
        self.config.write_text(json.dumps(config),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'signature'):generate(self.truth_args(resume=True))

    def test_strict_json_duplicate_names_and_quotes(self):
        with self.assertRaises(ValueError):strict_loads('{"x":1,"x":2}')
        with self.assertRaises(ValueError):strict_loads('{"x":NaN}')
        with self.assertRaises(ValueError):strict_loads('{"x":1e999}')
        sub=self.images/'sub';sub.mkdir();(sub/'MENU.PNG').write_bytes((self.images/'menu.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'duplicate'):scan_images(self.images)
        (sub/'MENU.PNG').unlink()
        t,p=self.build()
        p[0]['surprise']='ignored?'
        with self.assertRaises(ValueError):check_prediction(p[0])
        j=fake_hm.judge('summary');j['claims'][0]['quote']='not in candidate'
        with self.assertRaisesRegex(ValueError,'substring'):calculate(t,'summary',j,fake_hm.OUTPUTS['summary'])

    def test_reference_rejection_and_api_truncation(self):
        fake_hm.TRUNCATE_NEXT=True
        t,_=self.build()
        self.assertEqual(t['calls'][0]['attempts'],2)
        self.truth_path.unlink()
        fake_hm.REJECT_REFERENCE=True
        result=generate(self.truth_args())
        self.assertEqual(result['errors'],1)

    def test_prepare_only_and_tampered_score(self):
        result=generate(self.truth_args(prepare_only=True))
        self.assertEqual(len(fake_hm.CALLS),0)
        self.assertTrue(Path(result['requests']).exists())
        self.build();evaluate(self.eval_args())
        row=read_jsonl(self.eval_path)[0];row['score']=0.1
        with self.assertRaisesRegex(ValueError,'recalculation'):validate_result(row)
        self.assertEqual(char_count('甲 A\n😀！'),4)

    def test_explicit_rubric_and_external_evidence(self):
        t,_=self.build()
        ref=copy.deepcopy(t['reference'])
        del ref['interpretation']['fields'][0]['items'][0]['failure_conditions']
        with self.assertRaises(ValueError):check_reference(ref,t['context'])
        ev={'id':'external1','source':'external','location':'provided extract', 'text':'synthetic knowledge',
            'url':'https://example.org/fixture','version':'test-only'}
        ref=copy.deepcopy(t['reference']);ref['evidence'].append(ev)
        with self.assertRaisesRegex(ValueError,'supplied evidence'):validate_reference_for_input(ref,t)
        t['provided_evidence']=[copy.deepcopy(ev)];t['reference']=ref;t['reference_id']=reference_id(t)
        check_truth(t)
        t['reference']['evidence'][-1]['text']='fabricated replacement'
        with self.assertRaisesRegex(ValueError,'supplied evidence'):check_truth(t)

    def test_not_applicable_and_failed_prediction(self):
        t,p=self.build()
        t['reference']['applicability']['phrase']={'applicable':False,'reason':'test N/A'}
        t['reference_id']=reference_id(t);write_jsonl(self.truth_path,[t])
        p[0].update(status='error',output='',error='synthetic model outage')
        write_jsonl(self.pred_path,p);evaluate(self.eval_args())
        rows=read_jsonl(self.eval_path)
        self.assertEqual(rows[0]['evaluation_status'],'model_error')
        self.assertEqual(rows[0]['score'],0)
        self.assertEqual(rows[-1]['evaluation_status'],'not_applicable')
        self.assertIsNone(rows[-1]['score'])
        for r in rows:validate_result(r)
        self.assertEqual(aggregate(rows)['not_applicable'],1)

    def test_metadata_truncation_multirun_and_resume_tamper(self):
        _,p=self.build()
        p[0]['generation'].update(output_tokens=1024,finish_reason='length')
        second=copy.deepcopy(p[0]);second['run_id']='r2'
        write_jsonl(self.pred_path,p+[second]);evaluate(self.eval_args())
        rows=read_jsonl(self.eval_path)
        self.assertEqual(len(rows),8)
        self.assertEqual(sum(r['evaluation_status']=='missing_prediction' for r in rows),3)
        g=aggregate([rows[0]])['generation']
        self.assertEqual(g['truncation_rate_among_known'],1)
        self.assertEqual(g['known_output_tokens'],1)
        self.assertIsNone(aggregate([rows[1]])['generation']['truncation_rate_among_known'])
        self.assertTrue(rows[0]['warnings'])
        rows[0]['score']=0.1;write_jsonl(self.eval_path,rows)
        with self.assertRaisesRegex(ValueError,'recalculation'):evaluate(self.eval_args(resume=True))

    def test_cli_entrypoints_from_another_directory(self):
        _,p=self.build()
        truth=self.root/'cli-truth.jsonl';scores=self.root/'cli-scores.jsonl';report=self.root/'cli-report.html'
        env=os.environ.copy();env['PYTHONPATH']=str(HERE)
        commands=[['gen_truth.py',str(self.images),'--output',str(truth),'--api-config',str(self.api),'--task-config',str(self.config)],
            ['evaluate.py',str(truth),str(self.pred_path),'--output',str(scores),'--api-config',str(self.api)],
            ['get_html.py',str(scores),'--output',str(report)]]
        for script,*arguments in commands:
            completed=subprocess.run([sys.executable,str(HERE.parent/script),*arguments],cwd=self.root,env=env,
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertTrue(report.exists())
        self.assertIn('判错',report.read_text(encoding='utf-8'))

    def test_export_cannot_overwrite_image(self):
        image=self.images/'menu.png'
        before=image.read_bytes()
        with self.assertRaisesRegex(ValueError,'source image'):
            generate(self.truth_args(prepare_only=True,export_inputs=image))
        self.assertEqual(image.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
