(() => {
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const FILTERS = [['all', 'All'], ['happy', 'Happy flow'], ['ambiguity', 'Ambiguity'], ['edge', 'Edge cases'], ['passed', 'Passed'], ['failed', 'Failed'], ['error', 'Errored'], ['app', 'App chats']];
let runs = [], filter = 'all', query = '';

const kind = r => r.passed ? 'pass' : r.outcome === 'error' ? 'error' : 'fail';
const label = r => ({ pass: 'Passed', fail: 'Failed', error: 'Errored' })[kind(r)];
const when = t => { const d = new Date(t); return isNaN(d) ? '-' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ', ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }); };
// A test case code (EDGE-1) matches that case only, so EDGE-1 does not also bring EDGE-10; anything else is read as part of a session id or title.
const found = r => {
  const q = query.trim().toLowerCase(); if (!q) return true;
  if (/^[a-z]+-\d+$/.test(q)) return r.group !== 'app' && r.case_id.toLowerCase() === q;
  return (r.chat_id || '').toLowerCase().includes(q) || r.title.toLowerCase().includes(q);
};
const matches = r => found(r) && (filter === 'all' || r.group === filter || (filter === 'passed' && kind(r) === 'pass') || (filter === 'failed' && kind(r) === 'fail') || (filter === 'error' && kind(r) === 'error'));

function stats() {
  const n = k => runs.filter(r => kind(r) === k).length, box = $('#stats'); box.replaceChildren();
  [['Total runs', runs.length, ''], ['Passed', n('pass'), 'pass'], ['Failed', n('fail'), 'fail'], ['Errored', n('error'), 'error']].forEach(([name, value, cls]) => {
    const s = el('div', undefined, 'stat ' + cls); s.append(el('small', name), el('b', String(value))); box.append(s);
  });
}
function filters() {
  const box = $('#filters'); box.replaceChildren();
  FILTERS.forEach(([key, text]) => {
    const b = el('button', text, key === filter ? 'on' : ''); b.type = 'button'; b.setAttribute('aria-pressed', String(key === filter));
    b.addEventListener('click', () => { filter = key; filters(); rows(); }); box.append(b);
  });
}
function rows() {
  const body = $('#rows'), empty = $('#empty'), shown = runs.map((r, i) => ({ r, no: runs.length - i })).filter(x => matches(x.r));
  body.replaceChildren(); empty.hidden = shown.length > 0;
  empty.textContent = runs.length ? (query.trim() ? 'No run matches that search.' : 'No runs match this filter.') : 'No runs yet. Press Run on a test case, or Run all, and it will show up here.';
  shown.forEach(({ r, no }) => {
    const tr = el('tr'), c1 = el('td'), c3 = el('td'), c5 = el('td'), c8 = el('td');
    c1.append(el('span', String(no), 'num'));
    c3.append(el('span', r.title, 'ct'), el('span', r.group === 'app' ? r.case_id.slice(0, 8) : r.case_id.toUpperCase(), 'cid'));
    const bad = (r.checks || []).filter(k => !k.ok);
    if (bad.length && r.group !== 'app') { const ul = el('ul', undefined, 'why'); bad.slice(0, 4).forEach(k => ul.append(el('li', k.label + (k.detail ? ' (' + k.detail + ')' : '')))); c3.append(ul); }
    c5.append(el('span', label(r), 'badge ' + kind(r)));
    const held = (r.checks || []).filter(k => k.ok).length;
    if (r.chat_id) { const a = el('a', 'Open ', 'open'); a.href = '/chat/' + r.chat_id; a.append(el('span', '↗')); a.setAttribute('aria-label', `Open the chat of run ${no}: ${r.title}`); const t = el('a', 'Timeline', 'open'); t.href = '/timeline/' + r.chat_id; t.setAttribute('aria-label', `Open the timeline of run ${no}: ${r.title}`); const links = el('div', undefined, 'links'); links.append(a, t); c8.append(links); }
    tr.append(c1, el('td', when(r.at)), c3, el('td', r.group === 'app' ? 'App chat' : r.group === 'happy' ? 'Happy flow' : r.group === 'edge' ? 'Edge case' : 'Ambiguity'), c5, el('td', `${held} / ${(r.checks || []).length}`), el('td', r.seconds != null ? r.seconds + ' s' : '-'), c8);
    body.append(tr);
  });
}
async function load() {
  try { runs = await (await fetch('/api/runs')).json(); } catch (e) { runs = []; }
  try { runs = runs.concat(await (await fetch('/api/app-runs')).json()).sort((x, y) => new Date(y.at) - new Date(x.at)); } catch (e) { /* the app is offline: show the test runs only */ }
  stats(); filters(); rows();
}
$('#search').addEventListener('input', e => { query = e.target.value; rows(); });
load();
window.addEventListener('pageshow', e => { if (e.persisted) load(); });
})();
