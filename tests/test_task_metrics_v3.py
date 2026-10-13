"""按修订版文档核验任务专属公式、过滤分母、偏好比较及单文件交接。"""
import base64
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII=')

def load(name='evaluate.py'):
    spec = importlib.util.spec_from_file_location('task_metrics_' + name, ROOT / name)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev

def fixture(ev):
    digest = hashlib.sha256(PNG).hexdigest()
    prompts = {task: {'version':'vtest','task':task,'system':'按原图写中文正文','user':'请分析图片'} for task in ev.TASKS}
    image = {'schema_version':2,'dataset_id':'primary','run_id':'v3-run','image_id':'img001',
        'relative_path':'中文/实拍/menu.png','language':'中文','capture_type':'实拍','image_sha256':digest,
        'width':1,'height':1,'image_status':'ok','image_error':None,'duplicate_of':None,'prompts':prompts,
        'image':{'mime_type':'image/png','base64':base64.b64encode(PNG).decode('ascii'),'sha256':digest},
        'recommendation_requirement':{'id':'req01','text':'一人总预算50元','constraints':{'budget':{'amount':'50','currency':'CNY','scope':'total'},'diners':1,'hard_constraints':[],'preferences':[],'meal_requirements':{}},'fx_context':None}}
    texts = {'summary':'餐厅菜单\n牛肉面20元，蔬菜面15元。','interpretation':'菜单有牛肉面与蔬菜面。\n【菜品价格】\n- 牛肉面20元。\n- 蔬菜面15元。\n牛肉面比蔬菜面贵5元。','recommendation':'一人点一份牛肉面，单价20元，含全部费用共20元，符合50元预算。'}
    for task in ev.TASKS:
        image[task]={'status':'ok','text':texts[task],'raw_text':texts[task],'latency_seconds':.1,'image_sha256':digest,'prompt_sha256':ev.fingerprint(prompts[task]),'finish_reason':'stop','error':None}
    facts=[]
    for fid,cat,text,price in [('f001','name','牛肉面',None),('f002','price','牛肉面每份20元','20'),('f003','name','蔬菜面',None),('f004','price','蔬菜面每份15元','15')]:
        facts.append({'fact_id':fid,'category':cat,'source_text':text,'normalized_zh':text,'evidence':'菜单对应行名称及邻近价格','confidence':'high','importance':1,'required_for':['interpretation','recommendation'],
            'numeric':None if price is None else {'value':price,'unit':'份','currency':'CNY','qualifier':'每份'}})
    ref={'schema_version':3,'dataset_id':'primary','image_id':'img001','image_sha256':digest,'scene':'menu',
        'image_quality':{'readability':'clear','issues':[]},'unreadable_regions':[],'reference_status':'draft',
        'annotator':{'model':'mock','version':'v3'},'facts':facts,'requirement_sha256':ev.fingerprint(image['recommendation_requirement']),
        'recommendation_feasibility':{'requirement_id':'req01','status':'feasible','reason':'已知含全部费用，可点一份牛肉面20元','evidence_fact_ids':['f001','f002'],'fees_status':'complete','mandatory_fees':[]},
        'task_applicability':{task:{'status':'applicable','reason':'主体与餐品事实可辨','evidence_fact_ids':['f001']} for task in ev.TASKS},
        'interpretation_points':[{'point_id':'k001','kind':'core','description':'牛肉面','evidence_fact_ids':['f001'],'reasoning':'主体菜品'},
            {'point_id':'k002','kind':'core','description':'牛肉面20元','evidence_fact_ids':['f002'],'reasoning':'明确价格'},
            {'point_id':'k003','kind':'core','description':'蔬菜面','evidence_fact_ids':['f003'],'reasoning':'另一种菜品'},
            {'point_id':'e001','kind':'enrichment','description':'牛肉面贵5元','evidence_fact_ids':['f002','f004'],'reasoning':'20-15=5'}]}
    return image,ref

def score(ev,image,ref,task):
    row={'schema_version':3,'dataset_id':image['dataset_id'],'image_id':image['image_id'],'task':task,
        'candidate_sha256':ev.text_sha256(image[task]['text']),'reference_sha256':ev.fingerprint(ref),'judge_prompt_sha256':ev.text_sha256(ev.JUDGE_SYSTEM),
        'task_prompt_sha256':ev.fingerprint(image['prompts'][task]),'requirement_sha256':ev.fingerprint(image['recommendation_requirement']) if task=='recommendation' else None,
        'source_evidence_mode':'image_and_reference','scene':ref['scene'],'answer_kind':'content',
        'claims':[{'claim_id':'c001','text':'牛肉面','verdict':'supported','fact_ids':['f001'],'severity':'minor','explanation':'菜名可见','category':'name'},
            {'claim_id':'c002','text':'牛肉面20元','verdict':'supported','fact_ids':['f002'],'severity':'major','explanation':'价格可见','category':'price'}],
        'covered_fact_ids':[],'critical_errors':[],'needs_human_review':False,'review_reasons':[],'judge':{'model':'mock','version':'v3'},
        'entities':[],'point_checks':[],'unmatched_enrichments':[],'review_scope':[],'recommendation':None}
    if task=='summary':
        row['entities']=[{'entity_id':'n001','text':'牛肉面','verdict':'supported','fact_ids':['f001'],'reason':'菜单菜名'},
            {'entity_id':'n002','text':'蔬菜面','verdict':'supported','fact_ids':['f003'],'reason':'菜单菜名'}]
    if task=='interpretation':
        row['claims'].append({'claim_id':'c003','text':'牛肉面比蔬菜面贵5元','verdict':'supported','fact_ids':['f002','f004'],'severity':'minor','explanation':'20-15=5','category':'price'})
        for pid,cid in [('k001','c001'),('k002','c002'),('k003',None),('e001','c003')]:
            row['point_checks'].append({'point_id':pid,'status':'covered' if cid else 'missed','claim_ids':[cid] if cid else [],'accurate':bool(cid),'relevant':bool(cid),'adds_value':bool(cid),'nonredundant':bool(cid),'reason':'实际正确表达' if cid else '未覆盖'})
    if task=='recommendation':
        row['recommendation']={'requirement_id':'req01','disposition':'confirmed','requirement_checks':[{'constraint_id':cid,'status':'met','reason':'符合输入和实际方案','evidence_fact_ids':['f002']} for cid in ['budget','diners']],
            'plan':{'items':[{'name':'牛肉面','quantity':'1','unit_price':'20','currency':'CNY','price_fact_ids':['f002'],'person':'person_1'}], 'subtotal':'20','mandatory_fees':[],'total':'20','currency':'CNY','fx':None,'budget_total':'20','budget_currency':'CNY'},
            'refusal_assessment':None,'preference_checks':[],'better_alternative':None}
    return row

class TaskMetricsV3Tests(unittest.TestCase):
    def setUp(self):
        self.ev=load(); self.image,self.ref=fixture(self.ev)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def report(self, rows=None, refs=None, scores=None):
        return self.ev.build_report(rows or [self.image],refs or {'img001':self.ref},scores or {})
    def test_schemas_no_weights_or_direct_scores(self):
        self.ev.validate_reference(self.ref,self.image)
        self.assertNotIn('dimensions',self.ev.JUDGE_SCHEMA['properties'])
        self.assertNotIn('weight',json.dumps(self.ev.RUBRIC))
        for task in self.ev.TASKS:
            self.ev.validate_judgement(score(self.ev,self.image,self.ref,task),self.ref,self.image,task,'image_and_reference')
    def test_summary_accuracy_and_nouns_use_different_denominators(self):
        s=score(self.ev,self.image,self.ref,'summary')
        s['claims']+=[{**s['claims'][0],'claim_id':'c003','text':'牛肉面30元','verdict':'contradicted','fact_ids':['f002']}, {**s['claims'][0],'claim_id':'c004','text':'蔬菜面','fact_ids':['f003']}]
        s['entities'][1]['verdict']='contradicted'
        m=self.ev.score_judgement(s,self.ref)
        self.assertEqual(m['accuracy'],.75);self.assertEqual(m['noun_accuracy'],.5)
        self.assertEqual(m['fact_judged'],4);self.assertEqual(m['noun_judged'],2)
    def test_exact_duplicates_cannot_inflate_correctness(self):
        s=score(self.ev,self.image,self.ref,'summary')
        s['claims'][1].update(text='牛肉面30元',verdict='contradicted')
        s['claims'].append({**s['claims'][0],'claim_id':'c003'})
        self.assertEqual(self.ev.score_judgement(s,self.ref)['accuracy'],.5)
        with self.assertRaises(ValueError):self.ev.validate_judgement(s,self.ref,self.image,'summary','image_and_reference')
    def test_unknown_not_full_marks(self):
        s=score(self.ev,self.image,self.ref,'summary')
        for c in s['claims']:c['verdict']='unverifiable'
        for e in s['entities']:e['verdict']='unverifiable'
        s.update(needs_human_review=True,review_reasons=['无法核验'],review_scope=['facts'])
        m=self.ev.score_judgement(s,self.ref)
        self.assertIsNone(m['accuracy']);self.assertIsNone(m['noun_accuracy'])
    def test_empty_or_unsupported_refusal_on_readable_image_is_zero(self):
        for kind,text in [('empty',''),('refusal','不能分析这张图')]:
            self.image['summary']['text']=text
            s=score(self.ev,self.image,self.ref,'summary');s.update(answer_kind=kind,claims=[],entities=[])
            r=self.report(scores={('img001','summary'):s})['tasks']['summary']
            self.assertEqual(r['accuracy']['mean'],0);self.assertEqual(r['noun_accuracy']['mean'],0)
    def test_interpretation_formula_matches_document(self):
        s=score(self.ev,self.image,self.ref,'interpretation')
        m=self.ev.score_judgement(s,self.ref)
        self.assertEqual(m['content_score'],75);self.assertEqual(m['core_covered'],2);self.assertEqual(m['enrichment_covered'],1)
        self.assertEqual(m['core_total'],3);self.assertEqual(m['enrichment_total'],1)
    def test_bad_or_duplicate_content_cannot_score(self):
        s=score(self.ev,self.image,self.ref,'interpretation')
        for field in ['accurate','relevant','adds_value','nonredundant']:
            bad=copy.deepcopy(s);bad['point_checks'][-1][field]=False
            with self.assertRaises(ValueError):self.ev.validate_judgement(bad,self.ref,self.image,'interpretation','image_and_reference')
        s['point_checks'][-1]['claim_ids']=['c001']
        with self.assertRaises(ValueError):self.ev.validate_judgement(s,self.ref,self.image,'interpretation','image_and_reference')
    def test_unlisted_enrichment_is_pending_until_reference_updated(self):
        s=score(self.ev,self.image,self.ref,'interpretation');s['unmatched_enrichments']=[{'text':'另一条有用解释','claim_ids':['c003'],'reason':'需独立核验'}]
        s.update(needs_human_review=True,review_reasons=['先核验增量并统一修订参考'],review_scope=['content'])
        self.ev.validate_judgement(s,self.ref,self.image,'interpretation','image_and_reference')
        self.assertIsNone(self.ev.score_judgement(s,self.ref)['content_score'])
    def test_recommendation_correct_grade2_and_wrong_grade0(self):
        s=score(self.ev,self.image,self.ref,'recommendation')
        good=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual((good['recommendation_correctness'],good['recommendation_grade']),('correct',2))
        s['recommendation']['plan']['total']='10'
        bad=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual((bad['recommendation_correctness'],bad['recommendation_grade']),('incorrect',0))
    def test_wrong_source_price_currency_and_nonpositive_quantity_are_incorrect(self):
        for field,value in [('unit_price','21'),('currency','USD'),('quantity','0'),('quantity','-1')]:
            s=score(self.ev,self.image,self.ref,'recommendation');s['recommendation']['plan']['items'][0][field]=value
            self.ev.validate_judgement(s,self.ref,self.image,'recommendation','image_and_reference')
            self.assertEqual(self.ev.recommendation_result(s,self.image,self.ref)['recommendation_grade'],0)
    def test_missing_quantity_stays_unknown_instead_of_defaulting_to_one(self):
        s=score(self.ev,self.image,self.ref,'recommendation');s['recommendation']['plan']['items'][0]['quantity']=None
        self.ev.validate_judgement(s,self.ref,self.image,'recommendation','image_and_reference')
        self.assertEqual(self.ev.recommendation_result(s,self.image,self.ref)['recommendation_correctness'],'unknown')
    def test_grounded_infeasible_refusal_is_correct_despite_conflict_check(self):
        self.ref['recommendation_feasibility']['status']='infeasible'
        self.ref['recommendation_feasibility']['reason']='图上菜品均超50元预算'
        for index,price in [(1,'60'),(3,'70')]:
            fact=self.ref['facts'][index];fact['numeric']['value']=price
            fact['source_text']=fact['normalized_zh']=('牛肉面' if index==1 else '蔬菜面')+'每份'+price+'元'
        self.image['recommendation']['text']='图中两种面分别为60元和70元，均超过50元预算，无法提供符合要求的组合。'
        s=score(self.ev,self.image,self.ref,'recommendation');s['answer_kind']='refusal'
        s['claims']=[{**s['claims'][1],'text':'两种面60元与70元','fact_ids':['f002','f004']}]
        rec=s['recommendation'];rec.update(disposition='unable',plan=None,refusal_assessment={'justified':True,'reason':'准确指出图中全部可选单品超预算','fact_ids':['f002','f004'],'claim_ids':['c002']})
        rec['requirement_checks'][0]['status']='violated';rec['requirement_checks'][1]['status']='unknown'
        self.ev.validate_judgement(s,self.ref,self.image,'recommendation','image_and_reference')
        r=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual((r['recommendation_correctness'],r['recommendation_grade']),('correct',2))
        rec['refusal_assessment']['justified']=False
        self.assertEqual(self.ev.recommendation_result(s,self.image,self.ref)['recommendation_grade'],0)
    def with_preferences(self):
        self.image['recommendation_requirement']['constraints']['preferences']=['喜欢蔬菜面']
        self.ref['requirement_sha256']=self.ev.fingerprint(self.image['recommendation_requirement'])
        s=score(self.ev,self.image,self.ref,'recommendation')
        s['recommendation']['preference_checks']=[{'constraint_id':'p001','status':'violated','reason':'候选牛肉面不是偏好的蔬菜面','evidence_fact_ids':['f001','f003']}]
        alt=copy.deepcopy(s['recommendation']['plan'])
        alt['items'][0].update(name='蔬菜面',unit_price='15',price_fact_ids=['f004'])
        for key in ['subtotal','total','budget_total']:alt[key]='15'
        s['recommendation']['better_alternative']={'plan':alt,'requirement_checks':copy.deepcopy(s['recommendation']['requirement_checks']),
            'preference_checks':[{'constraint_id':'p001','status':'met','reason':'菜单蔬菜面可满足明确偏好','evidence_fact_ids':['f003']}],
            'reason':'同样满足硬约束，改善用户明示软偏好','evidence_fact_ids':['f003','f004']}
        return s
    def test_grade1_requires_verified_better_plan(self):
        s=self.with_preferences();self.ev.validate_judgement(s,self.ref,self.image,'recommendation','image_and_reference')
        r=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual(r['recommendation_grade'],1);self.assertTrue(r['alternative_check']['verified_better'])
    def test_invalid_alternative_does_not_fake_grade1(self):
        s=self.with_preferences();s['recommendation']['better_alternative']['plan']['total']='1'
        r=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual(r['recommendation_correctness'],'correct');self.assertIsNone(r['recommendation_grade'])
    def test_alternative_cannot_borrow_another_dish_price(self):
        s=self.with_preferences();s['recommendation']['better_alternative']['plan']['items'][0]['name']='未列出的菜品'
        r=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual(r['recommendation_correctness'],'correct');self.assertIsNone(r['recommendation_grade'])
        self.assertFalse(r['alternative_check']['verified_better'])
    def test_unknown_preference_does_not_prevent_hard_correctness(self):
        s=self.with_preferences();s['recommendation']['preference_checks'][0]['status']='unknown'
        s.update(needs_human_review=True,review_reasons=['偏好需要核验'],review_scope=['preferences'])
        r=self.ev.recommendation_result(s,self.image,self.ref)
        self.assertEqual(r['recommendation_correctness'],'correct');self.assertIsNone(r['recommendation_grade'])
    def test_unknown_hard_constraint_never_correct(self):
        self.image['recommendation_requirement']['constraints']['hard_constraints']=[{'id':'no_spicy'}]
        self.ref['requirement_sha256']=self.ev.fingerprint(self.image['recommendation_requirement'])
        self.ref['recommendation_feasibility']['status']='unknown'
        s=score(self.ev,self.image,self.ref,'recommendation');s['recommendation']['requirement_checks'].append({'constraint_id':'no_spicy','status':'unknown','reason':'图上未标辣度','evidence_fact_ids':[]})
        self.assertEqual(self.ev.recommendation_result(s,self.image,self.ref)['recommendation_correctness'],'unknown')
    def test_non_menu_excluded_and_food_poster_not_filtered_by_scene(self):
        self.ref['scene']='food'
        requests=self.ev.prepare_scores([self.image],{'img001':self.ref})
        self.assertEqual(len(requests),3)
        self.ref['task_applicability']['recommendation']={'status':'not_applicable','reason':'只有环境无可用餐品信息','evidence_fact_ids':[]}
        self.ref['recommendation_feasibility']['status']='not_applicable'
        requests=self.ev.prepare_scores([self.image],{'img001':self.ref})
        self.assertEqual(len(requests),2)
        r=self.report()['tasks']['recommendation'];self.assertEqual(r['excluded_images'],1);self.assertEqual(r['recommendation_denominator'],0)
    def test_correct_rate_filter_and_failure_denominator(self):
        s=score(self.ev,self.image,self.ref,'recommendation')
        second=copy.deepcopy(self.image);second['image_id']='img002';second['recommendation']['status']='error';second['recommendation']['error']='推理失败'
        ref2=copy.deepcopy(self.ref);ref2['image_id']='img002'
        r=self.report([self.image,second],{'img001':self.ref,'img002':ref2},{('img001','recommendation'):s})['tasks']['recommendation']
        self.assertEqual(r['recommendation_correct_rate'],.5);self.assertEqual(r['recommendation_denominator'],2)
    def test_hashes_and_empty_type_are_hard_checked(self):
        for field in ['candidate_sha256','reference_sha256','task_prompt_sha256','requirement_sha256','judge_prompt_sha256']:
            s=score(self.ev,self.image,self.ref,'recommendation');s[field]='0'*64
            with self.assertRaises(ValueError):self.ev.validate_judgement(s,self.ref,self.image,'recommendation','image_and_reference')
        s=score(self.ev,self.image,self.ref,'summary');s['answer_kind']='empty'
        with self.assertRaises(ValueError):self.ev.validate_judgement(s,self.ref,self.image,'summary','image_and_reference')
    def test_reference_points_require_independent_high_confidence(self):
        bad=copy.deepcopy(self.ref);bad['interpretation_points'][0]['evidence_fact_ids']=['missing']
        with self.assertRaises(ValueError):self.ev.validate_reference(bad,self.image)
        bad=copy.deepcopy(self.ref);bad['task_applicability']['summary']['status']='not_applicable'
        with self.assertRaises(ValueError):self.ev.validate_reference(bad,self.image)
    def test_html_displays_two_metrics_and_escapes_candidate(self):
        self.image['summary']['text']='<script>bad()</script>'
        r=self.report();page=self.ev.render_html(r)
        self.assertIn('&lt;script&gt;bad()',page);self.assertNotIn('<script>bad()',page)
        self.assertIn('逐图名词准确率均值',page);self.assertNotIn('综合质量分',page)
        self.assertNotIn('dimensions',self.ev.JUDGE_SCHEMA['properties'])
    def test_full_sdk_stages_single_file_resume_and_identical_formulas(self):
        hm=load('评估.py');image,ref=fixture(hm);calls=[]
        def managed(**kwargs):
            body=kwargs['req_data']; payload=json.loads(body['messages'][1]['content'][0]['text']); inputs=payload['input']
            self.assertEqual(body['messages'][0]['role'],'system')
            self.assertTrue(body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
            if 'candidate_text' not in inputs:
                result=copy.deepcopy(ref);result['annotator']=inputs['annotator'];calls.append('reference')
            else:
                result=score(hm,image,ref,inputs['task']);result['judge']=inputs['judge'];calls.append(inputs['task'])
            return {'choices':[{'message':{'content':json.dumps(result,ensure_ascii=False)},'finish_reason':'stop'}]}
        handoff={'schema_version':1,'artifact_type':'qwen_eval_handoff','datasets':[{'dataset_id':'primary','run_id':'v3-run','generation_results':[image]}]}
        target=self.root/'handoff.json';target.write_text(json.dumps(handoff),encoding='utf-8')
        with patch.object(hm,'DEFAULT_ACCOUNT','mock-account'),patch.object(hm,'DEFAULT_MODEL','mock'),patch.object(hm,'DEFAULT_MODEL_VERSION','v3'),patch.object(hm,'hm_aigc',SimpleNamespace(aigc_managed=managed),create=True):
            first=hm.main([str(target),'--stage','reference']);self.assertEqual(calls,['reference'])
            final=hm.main([str(target),'--stage','score']);self.assertEqual(calls,['reference','summary','interpretation','recommendation'])
            hm.main([str(target)]);self.assertEqual(len(calls),4)
        results=json.loads((self.root/'results.json').read_text(encoding='utf-8'))
        self.assertEqual(results['datasets']['primary']['tasks']['interpretation']['content_score']['mean'],75)
        self.assertEqual(results['datasets']['primary']['tasks']['recommendation']['recommendation_correct_rate'],1)
        self.assertTrue((self.root/'report.html').exists())
    def test_standalone_copy_without_site_packages_or_repository(self):
        script=self.root/'评估.py';script.write_bytes((ROOT/'评估.py').read_bytes())
        package={'schema_version':1,'artifact_type':'qwen_eval_handoff','datasets':[{'dataset_id':'primary','run_id':'v3-run','generation_results':[self.image]}]}
        target=self.root/'handoff.json';target.write_text(json.dumps(package),encoding='utf-8')
        result=subprocess.run([sys.executable,'-S',str(script),str(target),'--prepare-only'],cwd=self.root,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'));self.assertTrue((self.root/'report.html').exists())

if __name__=='__main__':unittest.main()
