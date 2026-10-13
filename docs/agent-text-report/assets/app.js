'use strict';
const D = window.REPORT_DATA;
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {paper:'菜单翻译',food:'菜品推荐',ticket:'票务推荐',interactive:'交互问答'};
const languages = {English:'英语',French:'法语',Spanish:'西班牙语'};
const fmt = value => Number.isInteger(value) ? String(value) : Number(value).toFixed(2);
const money = value => value == null ? '未知' : '€ '+(value/100).toFixed(2);
const spicy = value => ({yes:'辣',no:'不辣',unknown:'未知'}[value] || '未知');
let group = 'all', visible = [], selected = null;

function description(r) {
  if(r.case_id === 'F03') return '方案满足约束，但原始说明把48欧误写成78欧；百分制未覆盖这个解释错误。';
  if(r.group==='paper') return `${languages[r.language]} · 6 项菜单。${r.raw_unsupported_spicy.length} 项原始辣度标记需纠正，另列完整译文与价格。`;
  if(r.group==='food') return `预算 ${money(r.constraints.budget_minor)} · ${r.constraints.people} 人。${r.attempts.length>1 ? r.attempts.length+' 次推荐尝试，含超预算修正。' : '首次方案符合硬约束。'}`;
  if(r.group==='ticket') return `${r.ticket.adults} 位成人，儿童年龄 ${r.ticket.ages.length?r.ticket.ages.join('、')+' 岁':'无'}；最终含手续费 ${money(r.ticket.total_minor)}。`;
  return r.answers.length ? `${r.questions.length} 次提问，${r.answers.length} 次预置答复；查看条件保存与最终结果。` : '无需追问；检查只翻译、保存条件或未知价格处理。';
}

function card(r) {
  const score = r.final ? `<b>${fmt(r.first_draft.score_100)}</b><small>首次</small><span class="arrow">→</span><b class="final">${fmt(r.final.score_100)}</b><small>最终 / 100</small>` : '<b class="pass-score">通过</b><small>既有独立审查 · 未设百分制</small>';
  const badge = r.needs_correction ? '<span class="tag warning">含修正过程</span>' : '<span class="tag">最终通过</span>';
  return `<button class="case-card" data-case="${r.case_id}" aria-label="查看 ${r.case_id} ${escapeHTML(r.title)}"><div class="card-top"><span class="case-id">${r.case_id} · ${labels[r.group]}</span>${badge}</div><h3>${escapeHTML(r.title)}</h3><p>${escapeHTML(description(r))}</p><div class="card-score">${score}</div><div class="card-footer"><span>最终回复 ${r.final_reply_chars} 字符</span><span>查看完整样例 ↗</span></div></button>`;
}

function render() {
  const term = $('search').value.trim().toLocaleLowerCase();
  visible = D.records.filter(r => (group==='all'||r.group===group) && (!$('featured').checked||r.featured) &&
    ($('language').value==='all'||r.language===$('language').value) && (!$('corrected').checked||r.needs_correction) &&
    (!term||[r.case_id,r.title,r.prompt,r.final_reply,r.first_draft_notes,r.last_raw_model_reply].join(' ').toLocaleLowerCase().includes(term)))
    .sort((a,b)=>['paper','food','ticket','interactive'].indexOf(a.group)-['paper','food','ticket','interactive'].indexOf(b.group)||a.case_id.localeCompare(b.case_id));
  $('case-grid').innerHTML=visible.map(card).join('');
  $('case-count').textContent=`展示 ${visible.length} / 45 个样例${$('featured').checked?' · 精选模式':''}`;
  $('empty').hidden=visible.length>0;
  $('case-grid').querySelectorAll('[data-case]').forEach(b => b.addEventListener('click',()=>openCase(b.dataset.case)));
}

function source(r) {
  let top = r.source_image ? `<img src="${r.source_image}" alt="${r.case_id} 原始${languages[r.language]}菜单" loading="lazy"><a class="source-link" href="${r.source_image}" target="_blank" rel="noopener">打开原图 ↗</a>` : '';
  const c=r.constraints||{};
  const chips=[];
  if(c.people)chips.push(c.people+' 人');
  if(c.budget_minor!=null)chips.push('预算 '+money(c.budget_minor));
  if(c.no_spicy)chips.push('不吃辣');
  if(c.allergies?.length)chips.push('过敏：'+c.allergies.join('、'));
  if(r.ticket)chips.push(r.ticket.adults+' 成人',r.ticket.ages.length?'儿童 '+r.ticket.ages.join('、')+' 岁':'无儿童');
  top += `<div>${chips.map(s=>`<span class="detail-chip">${escapeHTML(s)}</span>`).join('')}</div>`;
  if(r.group==='paper'||r.group==='interactive')top+=`<p>${escapeHTML(r.prompt.length<220?r.prompt:r.prompt.slice(0,150)+'…（下方展开全文）')}</p>`;
  top+=`<details><summary>展开完整原始输入</summary><pre>${escapeHTML(r.prompt)}</pre></details>`;
  if(r.group==='food')top+=`<details><summary>查看给定菜单与单价</summary><div class="table-wrap"><table class="comparison-table"><thead><tr><th>菜品</th><th>价格</th><th>辣度</th></tr></thead><tbody>${r.menu.map(x=>`<tr><td>${escapeHTML(x.translated)}<br>${escapeHTML(x.original)}</td><td>${money(x.price_minor)}</td><td>${spicy(x.spicy)}</td></tr>`).join('')}</tbody></table></div></details>`;
  if(r.answers.length)top+=`<h4>实际提问与预置答复</h4>${r.answers.map(a=>`<div class="qa"><b>问</b><p>${escapeHTML(a.question)}</p><b>答</b><p>${escapeHTML(a.answer)}</p></div>`).join('')}`;
  return `<aside class="input-panel"><h3>输入与条件</h3>${top}<div class="detail-note">${r.answers.length?'答复来自原实验预置输入':'输入来自原实验记录'}；本页不调用模型。</div></aside>`;
}

function scoreBand(r) {
  if(!r.final)return `<div class="detail-score"><div class="score-pair"><div><span>既有独立审查</span><b>通过</b></div></div><p>本次百分制只覆盖菜单翻译与菜品推荐。这一例展示原任务的通过性审查、文本与长度，不补造百分制分数。</p></div>`;
  const intro=r.group==='paper'?'菜名40 + 覆盖20 + 价格20 + 未知信息20。逐项按正确比例计分。':'菜品与数量30 + 已给忌口30 + 预算40。此分数不包含说明文字的算术、偏好或份量评估。';
  return `<div class="detail-score"><div class="score-pair"><div><span>原始首次</span><b>${fmt(r.first_draft.score_100)}</b></div><div class="to">→</div><div><span>最终保存</span><b>${fmt(r.final.score_100)}</b></div></div><p>${intro}<br>满分100；仅表示检查项符合。</p></div>`;
}

function scoreTable(r) {
  if(!r.final)return '';
  let rows;
  if(r.group==='paper')rows=[['菜名含义',40,'semantic_correct_items'],['完整覆盖',20,'exact_coverage'],['价格与币种',20,'price_currency_correct_items'],['未知信息处理',20,'uncertainty_correct_items']].map(([name,w,key])=>{
    const v=x=>key==='exact_coverage'?(x[key]?w:0):w*x[key]/6;
    return [name+' / '+w,fmt(v(r.first_draft)),fmt(v(r.final))];
  });
  else rows=[['菜品及数量',30,'items_quantity_legal'],['已给忌口',30,'stated_diet_constraints_met'],['预算',40,'budget_met']].map(([n,w,k])=>[n+' / '+w,r.first_draft[k]?w:0,r.final[k]?w:0]);
  return `<details><summary>查看各维度得分</summary><div class="table-wrap"><table class="comparison-table"><thead><tr><th>维度 / 权重</th><th>原始首次</th><th>最终</th></tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`;
}

function translation(r) {
  const rows=r.menu.map(x=>{
    const raw=r.first_menu.find(y=>y.original===x.original);const changed=raw.spicy!==x.spicy;
    return `<tr><td>${escapeHTML(x.original)}</td><td>${escapeHTML(raw.translated)}</td><td>${escapeHTML(x.translated)}</td><td>${money(x.price_minor)}</td><td>${changed?`<span class="changed">${spicy(raw.spicy)} → ${spicy(x.spicy)}</span>`:spicy(x.spicy)}</td></tr>`;
  }).join('');
  return `<section class="detail-block"><h3>逐项翻译与属性对照</h3><div class="table-wrap"><table class="comparison-table"><thead><tr><th>原文</th><th>首次译文</th><th>最终译文</th><th>价格</th><th>辣度：首次 → 最终</th></tr></thead><tbody>${rows}</tbody></table></div><p class="detail-note">橙色为工程纠正；未知价格保留未知，不能凭菜名确认未标明的辣度。</p>${scoreTable(r)}</section>`;
}

function recommendation(r) {
  const list=picks=>picks.map(p=>{
    const item=r.menu.find(x=>x.id===p.id);
    return `${p.quantity} × ${item?.translated||p.id} · ${money((item?.price_minor||0)*p.quantity)}`;
  }).join('；');
  const timeline=r.attempts.map((a,i)=>`<div class="timeline-item"><b>第 ${i+1} 次提交 · <span class="${a.error?'changed':'correct'}">${a.error?'被校验拒绝':'通过校验'}</span></b><p>${escapeHTML(list(a.picks))}</p><p>实际合计 ${money(a.check.total_minor)} / 预算 ${money(a.check.budget_minor)}</p>${a.error?`<p>${escapeHTML(a.error)}</p>`:''}<details><summary>当次原始说明</summary><pre>${escapeHTML(a.notes)}</pre></details></div>`).join('');
  const special=r.case_id==='F03'?'<div class="explanation warning">这一例首次方案的硬约束分为100，但原始说明把实际48欧写成78欧、误称超预算。说明文字的错误未包含在硬约束分中，必须另外核验。</div>':'';
  return `<section class="detail-block"><h3>推荐尝试与修正</h3>${special}${timeline}${scoreTable(r)}</section>`;
}

function ticket(r) {
  if(!r.ticket)return '';
  const t=r.ticket;
  return `<section class="detail-block"><h3>最终票型与费用核验</h3><div class="explanation">${t.adults} 位成人；儿童年龄 ${t.ages.length?t.ages.join('、'):'无'}。最终选择 ${t.kind==='family'?'家庭票':'按人数组合票'}，含手续费 ${money(t.total_minor)}。</div><div class="table-wrap"><table class="comparison-table"><thead><tr><th>合法方案</th><th>票价</th><th>手续费</th><th>合计</th></tr></thead><tbody>${t.valid_options.map(x=>`<tr><td>${x.kind==='family'?'家庭票':'按人数组合票'}${x.kind===t.kind?' ✓':''}</td><td>${money(x.subtotal_minor)}</td><td>${money(x.fee_minor)}</td><td>${money(x.total_minor)}</td></tr>`).join('')}</tbody></table></div><p class="detail-note">这里只评票种推荐和文字说明，不包含设备操作或出票成功率。</p></section>`;
}

function openCase(id) {
  selected=D.records.find(x=>x.case_id===id);
  if(!selected)return;
  const r=selected;
  $('detail-title').textContent=r.title;
  $('detail-kicker').textContent=`${r.case_id} / ${labels[r.group]}${r.language?' / '+languages[r.language]:''}`;
  const context=r.group==='paper'?translation(r):r.group==='food'?recommendation(r):ticket(r);
  const final=`<section class="detail-block final-block"><div class="detail-label">最终展示 · ${r.final_reply_chars} 字符</div><h3>用户看到的最终回复</h3><pre>${escapeHTML(r.final_reply)}</pre>${r.group==='paper'?'<p class="detail-note">这里只是回复摘要；完整译文见上方六项菜单。</p>':''}</section>`;
  const raw=`<section class="detail-block raw-block"><div class="detail-label">末轮原始回复 · ${r.last_raw_reply_chars} 字符</div><h3>工程整理前的自然语言输出</h3><pre>${escapeHTML(r.last_raw_model_reply)}</pre><p class="detail-note">这是末轮模型回复，可能已经历反馈与重试，不等同于首次方案。与上方最终回复的差异包含程序整理。</p></section>`;
  const evidence=`<section class="detail-block"><h3>复核与长度记录</h3><p class="audit-note">${escapeHTML(r.independent_review_reason)}</p><dl class="numbers"><div><dt>模型调用</dt><dd>${r.model_calls} 次</dd></div><div><dt>累计输出</dt><dd>${r.output_tokens_all_calls} tokens</dd></div><div><dt>单次输出峰值</dt><dd>${r.max_single_call_output_tokens} / 4096 tokens</dd></div><div><dt>输出上限截断</dt><dd>${r.finish_reasons.length||0} 次</dd></div></dl><details><summary>原始结果完整性指纹</summary><pre>${r.source_sha256}</pre></details></section>`;
  $('detail-content').innerHTML=`<div class="detail-main">${scoreBand(r)}<div class="detail-columns">${source(r)}<div>${context}${final}${raw}${evidence}</div></div></div>`;
  const i=visible.findIndex(x=>x.case_id===id);
  $('detail-position').textContent=`${i+1} / ${visible.length} · 当前筛选结果`;
  $('previous').disabled=i<=0;$('next').disabled=i<0||i>=visible.length-1;
  if(!$('case-dialog').open)$('case-dialog').showModal();
  document.body.classList.add('dialog-open');$('case-dialog').scrollTop=0;
}

document.querySelectorAll('[data-group]').forEach(b=>b.addEventListener('click',()=>{
  group=b.dataset.group;
  document.querySelectorAll('[data-group]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});
  render();
}));
['featured','language','corrected'].forEach(id=>$(id).addEventListener('change',render));
$('search').addEventListener('input',render);
$('close-dialog').addEventListener('click',()=>$('case-dialog').close());
$('case-dialog').addEventListener('close',()=>document.body.classList.remove('dialog-open'));
$('case-dialog').addEventListener('click',e=>{if(e.target===$('case-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
for(const [id,delta] of [['previous',-1],['next',1]])$(id).addEventListener('click',()=>{
  const i=visible.findIndex(x=>x.case_id===selected?.case_id);if(visible[i+delta])openCase(visible[i+delta].case_id);
});
$('print').addEventListener('click',()=>window.print());
$('length-chart').innerHTML=Object.entries(D.summary.lengths).map(([key,x])=>{
  const raw=x.last_raw_reply_chars.mean,final=x.final_reply_chars.mean;
  return `<div class="bar-group"><div class="bar-label"><b>${labels[key]}</b><small>n = ${x.final_reply_chars.n}</small></div><div class="bar-row"><div class="bar-track"><div class="bar-fill" style="width:${raw/300*100}%"></div></div><span>${raw.toFixed(1)}</span></div><div class="bar-row"><div class="bar-track"><div class="bar-fill final" style="width:${final/300*100}%"></div></div><span>${final.toFixed(1)}</span></div></div>`;
}).join('');
render();
