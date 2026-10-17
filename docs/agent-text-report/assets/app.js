'use strict';
const data = window.REPORT_DATA;
const labels = {paper:'菜单翻译',food:'菜品推荐',ticket:'票务推荐',interactive:'补充信息问答'};
const languageNames = {English:'英语',French:'法语',Spanish:'西班牙语'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => Number.isInteger(value) ? String(value) : Number(value).toFixed(2);
const money = value => value == null ? '未知' : `${num(value / 100)} 欧元`;
const spice = value => ({yes:'辣',no:'不辣',unknown:'未知'}[value] || '未知');
const table = (heads, rows, cls='') => `<div class="table-wrap"><table class="${cls}"><thead><tr>${heads.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
function scoring(r) {
  if (!r.final) return '<p class="result"><b>结果：通过既有独立审查。</b>本类未设百分制。</p>';
  const f = r.first_draft;
  const math = r.group === 'paper'
    ? `${num(f.semantic_correct_items / 6 * 40)} + ${f.exact_coverage ? 20 : 0} + ${num(f.price_currency_correct_items / 6 * 20)} + ${num(f.uncertainty_correct_items / 6 * 20)}`
    : `${f.items_quantity_legal ? 30 : 0} + ${f.stated_diet_constraints_met ? 30 : 0} + ${f.budget_met ? 40 : 0}`;
  return `<p class="result"><b>模型初稿 ${num(f.score_100)} 分 → 系统最终方案 ${num(r.final.score_100)} 分。</b><br>首次计分：${math} = ${num(f.score_100)}；各项顺序与上方量表一致。</p>`;
}
function translation(r) {
  const rows = r.menu.map(item => {
    const first = r.first_menu.find(x => x.original === item.original);
    const changed = first.spicy !== item.spicy;
    return [esc(item.original),esc(first.translated),esc(item.translated),money(item.price_minor),changed ? `<span class="changed">${spice(first.spicy)} → ${spice(item.spicy)}</span>` : spice(item.spicy)];
  });
  return `<p><b>输入：</b>${esc(r.prompt)}（${languageNames[r.language]}）</p>${scoring(r)}
    ${table(['原文','首次译文','最终译文','价格','辣度：首次 → 最终'],rows,'sample-table')}
    <p class="small">${r.raw_unsupported_spicy.length} 项辣度由“不辣”改为“未知”：原图未注明，不能凭菜名判断。</p>
    <details><summary>查看菜单原图</summary><img src="${esc(r.source_image)}" alt="${r.case_id} ${languageNames[r.language]}菜单原图" loading="lazy"></details>`;
}
function food(r) {
  const c = r.constraints;
  const diet = [c.no_spicy ? '不吃辣' : '', c.allergies?.length ? `过敏：${c.allergies.map(x=>x==='shrimp'?'虾':x).join('、')}` : ''].filter(Boolean).join('；') || '无额外忌口';
  const picks = list => list.map(p => `${esc(r.menu.find(m=>m.id===p.id)?.translated || p.id)} × ${p.quantity}`).join(' + ');
  const tries = r.attempts.map((a,i)=>`<li>第 ${i+1} 次：${picks(a.picks)}，合计 <b>${money(a.check.total_minor)}</b>；${a.error?'超预算，程序拒绝后重试':'符合条件'}。</li>`).join('');
  const explanation = r.case_id === 'F03' ? '<p class="result"><b>说明文字另有错误：</b>方案实际 48 欧元，首次说明写成 78 欧元。方案分为 100，但说明仍需程序修正。</p>' : '';
  return `<p><b>输入：</b>${c.people} 人，预算 ${money(c.budget_minor)}；${esc(diet)}。菜单价格与属性已给定。</p>${scoring(r)}<ol class="attempts">${tries}</ol>${explanation}
    <details><summary>查看给定菜单和首次说明</summary>${table(['菜品','单价','已给辣度'],r.menu.map(m=>[esc(m.translated),money(m.price_minor),spice(m.spicy)]))}<pre>${esc(r.first_draft_notes)}</pre></details>`;
}
function ticket(r) {
  if (!r.ticket) return '';
  const t = r.ticket;
  const options = t.valid_options.map(x=>[x.kind==='family'?'家庭票':'按人数组合票',money(x.subtotal_minor),money(x.fee_minor),money(x.total_minor)]);
  return `<p><b>同行：</b>${t.adults} 位成人；${t.ages.length ? `儿童 ${t.ages.join('、')} 岁` : '无儿童'}。</p>
    <p class="small">票规：成人 24 欧元；6–17 岁儿童 12 欧元，未满 6 岁免费；2 成人 + 2 名付费儿童可买 60 欧元家庭票；每单另收 2 欧元。</p>
    ${table(['合法方案','票价','手续费','合计'],options)}`;
}
function interactive(r) {
  const qa = r.answers.map(a=>`<p><b>助手问：</b>${esc(a.question)}</p><p><b>预置答复：</b>${esc(a.answer)}</p>`).join('');
  const input = r.case_id==='Q02' ? '为两位成人和两位儿童选经济票种，缺少儿童年龄。' : r.prompt;
  const menu = r.menu.length ? table(['原文','最终译文','价格'],r.menu.map(m=>[esc(m.original),esc(m.translated),money(m.price_minor)]),'sample-table') : '';
  return `<p><b>输入：</b>${esc(input)}</p>${qa || '<p>信息已足够，本例没有追问。</p>'}${menu}${ticket(r)}`;
}
function showCase(group, id) {
  const r = data.records.find(x=>x.case_id===id && x.group===group);
  if (!r) return;
  let content;
  if (group==='paper') content=translation(r);
  else if (group==='food') content=food(r);
  else content=(group==='ticket'?'<p><b>输入：</b>按给定票规和同行条件推荐最便宜的合法方案，计入手续费。</p>'+ticket(r):interactive(r))+scoring(r);
  const ticketDraft = r.ticket_attempts?.[0]?.args?.reason;
  document.getElementById(`sample-${group}`).innerHTML=`${content}
    <h4>最终回复原文 <span class="small">· ${r.final_reply_chars} 字符</span></h4><pre>${esc(r.final_reply)}</pre>
    ${group==='paper'?'<p class="small">完整译文在上方菜单表内；本段只是随菜单展示的回复。</p>':''}
    <details><summary>查看原始记录</summary><h4>完整输入</h4><pre>${esc(r.prompt)}</pre>${ticketDraft?`<h4>首次票务说明</h4><pre>${esc(ticketDraft)}</pre>`:''}<h4>模型末轮回复 · ${r.last_raw_reply_chars} 字符</h4><p class="small">该回复可能已经过重试，不等于首次方案。</p><pre>${esc(r.last_raw_model_reply)}</pre><h4>既有独立审查意见</h4><p>${esc(r.independent_review_reason)}</p></details>`;
}
const defaults = {paper:'P01',food:'F01',ticket:'T01',interactive:'Q02'};
 document.getElementById('examples-list').innerHTML=Object.entries(labels).map(([group,label])=>{
  const records=data.records.filter(r=>r.group===group).sort((a,b)=>a.case_id.localeCompare(b.case_id));
  return `<article class="example"><div class="example-head"><h3>${label}</h3><label><span class="sr-only">切换${label}样例</span><select id="select-${group}">${records.map(r=>`<option value="${r.case_id}"${r.case_id===defaults[group]?' selected':''}>${r.case_id} · ${esc(r.title)}</option>`).join('')}</select></label></div><div id="sample-${group}" aria-live="polite"></div></article>`;
}).join('');
for (const group of Object.keys(labels)) {
  document.getElementById(`select-${group}`).addEventListener('change',e=>showCase(group,e.target.value));
  showCase(group, defaults[group]);
}
document.getElementById('length-rows').innerHTML=Object.entries(data.summary.lengths).map(([group,s])=>`<tr><th>${labels[group]}（${s.final_reply_chars.n} 例）</th><td>${s.last_raw_reply_chars.mean.toFixed(1)}</td><td>${s.final_reply_chars.mean.toFixed(1)}</td><td>${s.final_reply_chars.max}</td></tr>`).join('');
