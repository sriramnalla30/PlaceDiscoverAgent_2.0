(() => {
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const FILTERS = [['all', 'All'], ['not_run', 'Not run'], ['pass', 'Passed'], ['fail', 'Failed'], ['review', 'Needs review'], ['no_calls', 'No scout call']];
const LABEL = { pass: 'Pass', fail: 'Fail', review: 'Review', unknown: 'Not fully checked', no_calls: 'No scout call' };
const TONE = { pass: 'pass', fail: 'fail', review: 'error', unknown: '' };
let evals = [], rows = [], runNo = {}, focus = 'all', filter = 'all', query = '', timer = null, openCell = null, lastFocus = null;

const when = t => { const d = new Date(t * 1000); return isNaN(d) ? '-' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ', ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }); };
const calls = (n, unit = 'scout call') => n + ' ' + unit + (n === 1 ? '' : 's');
// what a cell counts as, for the filters and the counts: not_run, busy, or the verdict
const keyOf = c => c.state === 'done' ? c.verdict : c.state === 'not_run' ? 'not_run' : 'busy';
const evalOf = id => evals.find(e => e.id === id);

// With one eval in focus a session matches by that eval's cell. With "All evals" it matches when any eval's cell has that result
// (Passed needs a result in every eval that was run and no failure).
function matches(r) {
  if (filter === 'all') return true;
  const cells = focus === 'all' ? evals.map(e => r.cells[e.id]) : [r.cells[focus]], keys = cells.map(keyOf);
  if (filter === 'pass') return keys.some(k => k === 'pass') && keys.every(k => k === 'pass' || k === 'not_run');
  return keys.includes(filter);
}
const found = r => { const q = query.trim().toLowerCase(); return !q || r.id.toLowerCase().includes(q) || r.title.toLowerCase().includes(q); };

function counts(id) {
  const n = {}; rows.forEach(r => { const k = keyOf(r.cells[id]); n[k] = (n[k] || 0) + 1; }); return n;
}
function cards() {
  const box = $('#evcards'); box.replaceChildren();
  const all = el('button', undefined, 'evcard' + (focus === 'all' ? ' on' : '')); all.type = 'button'; all.setAttribute('aria-pressed', String(focus === 'all'));
  all.append(el('span', 'ALL', 'tag'), el('h3', 'All evals'), el('p', rows.length + ' sessions · ' + evals.length + ' evals', 'sub'));
  all.addEventListener('click', () => { focus = 'all'; draw(); }); box.append(all);
  evals.forEach(e => {
    const n = counts(e.id), card = el('div', undefined, 'evcard' + (focus === e.id ? ' on' : ''));
    const pick = el('button', undefined, 'evpick'); pick.type = 'button'; pick.setAttribute('aria-pressed', String(focus === e.id));
    pick.append(el('span', 'EVAL ' + e.num, 'tag'), el('h3', e.title), el('p', e.subtitle, 'sub'));
    pick.addEventListener('click', () => { focus = e.id; draw(); });
    const line = el('div', undefined, 'evnums');
    [['pass', n.pass || 0, 'pass'], ['fail', n.fail || 0, 'fail'], ['review', (n.review || 0) + (n.unknown || 0), 'error'], ['not run', n.not_run || 0, '']].forEach(([name, v, cls]) => {
      const s = el('span', undefined, 'evnum ' + cls); s.append(el('b', String(v)), document.createTextNode(' ' + name)); line.append(s);
    });
    card.append(pick, line);
    const todo = rows.filter(r => r.cells[e.id].state === 'not_run').map(r => r.id);
    if (todo.length) {
      const b = el('button', 'Run ' + todo.length + ' not run', 'linkbtn'); b.type = 'button';
      b.addEventListener('click', () => { if (confirm('Run "' + e.title + '" on ' + todo.length + ' sessions? Each one calls the judge model.')) run(e.id, todo); });
      card.append(b);
    }
    box.append(card);
  });
}
function filters() {
  const box = $('#filters'); box.replaceChildren();
  FILTERS.forEach(([k, text]) => {
    const b = el('button', text, k === filter ? 'on' : ''); b.type = 'button'; b.setAttribute('aria-pressed', String(k === filter));
    b.addEventListener('click', () => { filter = k; filters(); draw(); }); box.append(b);
  });
}
async function run(evalId, ids) {
  const r = await fetch('/api/eval/' + evalId + '/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_ids: ids }) });
  if (!r.ok) { alert('Could not start the eval.'); return; }
  await load();
}

// ---- the table: one compact result per eval ----
function head() {
  const tr = el('tr'); tr.append(el('th', 'Session', 'stick'));
  evals.forEach(e => {
    const th = el('th', undefined, 'evcol' + (focus === e.id ? ' on' : '')), b = el('button', undefined, 'colbtn'); b.type = 'button';
    b.append(el('span', 'EVAL ' + e.num, 'cn'), el('span', e.title, 'ct2')); b.addEventListener('click', () => { focus = focus === e.id ? 'all' : e.id; draw(); });
    th.append(b); tr.append(th);
  });
  $('#head').replaceChildren(tr);
}
function chip(r, e) {
  const c = r.cells[e.id], td = el('td', undefined, 'evcell' + (focus !== 'all' && focus !== e.id ? ' dim' : ''));
  if (c.state === 'not_run') {
    const b = el('button', 'Run', 'run small'); b.type = 'button'; b.setAttribute('aria-label', 'Run ' + e.title + ' on this session');
    b.addEventListener('click', () => run(e.id, [r.id])); td.append(el('span', 'Not run', 'badge'), b);
  } else if (c.state === 'queued' || c.state === 'running') {
    td.append(el('span', c.state === 'queued' ? 'Waiting…' : 'Running…', 'badge busy'));
  } else {
    const b = el('button', LABEL[c.verdict] || c.verdict, 'badge chipbtn ' + (TONE[c.verdict] || '')); b.type = 'button';
    b.setAttribute('aria-label', e.title + ': ' + (LABEL[c.verdict] || c.verdict) + '. Show details');
    b.addEventListener('click', ev => openDrawer(r.id, e.id, ev.currentTarget));
    td.append(b);
    if (c.count) td.append(el('span', String(c.count), 'cid cnt'));
  }
  if (c.error) { const w = el('div', 'Last run failed', 'evline why'); w.title = c.error; td.append(w); }
  return td;
}
function draw() {
  cards(); head();
  const list = rows.filter(r => found(r) && matches(r)), body = $('#rows'); body.replaceChildren();
  list.forEach(r => {
    const tr = el('tr'), s = el('td', undefined, 'stick');
    s.append(el('span', r.title, 'ct'));
    const meta = el('span', undefined, 'cid'); meta.append(document.createTextNode((runNo[r.id] ? 'Run ' + runNo[r.id] + ' · ' : '') + r.id + ' · ' + when(r.updated) + ' · '));
    const a = el('a', 'Open'); a.href = '/timeline/' + r.id; a.target = '_blank'; a.rel = 'noopener'; meta.append(a); s.append(meta);
    tr.append(s); evals.forEach(e => tr.append(chip(r, e))); body.append(tr);
  });
  $('#count').textContent = list.length + ' of ' + rows.length + ' sessions';
  const empty = $('#empty'); empty.hidden = list.length > 0; empty.textContent = 'No session matches.';
  if (openCell) renderDrawer();
}

// ---- the drawer: what one eval checked in one session ----
function openDrawer(sid, evalId, trigger) { openCell = { sid, evalId }; lastFocus = trigger; renderDrawer(); }
function closeDrawer() { openCell = null; $('#drawer').hidden = true; $('#scrim').hidden = true; if (lastFocus && document.contains(lastFocus)) lastFocus.focus(); }
function renderDrawer() {
  const r = rows.find(x => x.id === openCell.sid), e = evalOf(openCell.evalId), d = $('#drawer');
  if (!r || !e) { closeDrawer(); return; }
  const c = r.cells[e.id], wasHidden = d.hidden;
  d.replaceChildren();
  const top = el('div', undefined, 'dtop'), who = el('div');
  who.append(el('div', 'EVAL ' + e.num + ' · ' + e.title, 'eyebrow'), el('h2', r.title, 'dh'), el('div', (runNo[r.id] ? 'Run ' + runNo[r.id] + ' · ' : '') + r.id, 'cid'));
  const x = el('button', '×', 'dclose'); x.type = 'button'; x.setAttribute('aria-label', 'Close details'); x.addEventListener('click', closeDrawer);
  top.append(who, x); d.append(top);
  d.querySelector('h2').id = 'dtitle';
  const bar = el('div', undefined, 'dbar');
  if (c.state === 'done') {
    bar.append(el('span', LABEL[c.verdict] || c.verdict, 'badge ' + (TONE[c.verdict] || '')), el('span', calls(c.count, e.unit || undefined) + ' · run ' + when(c.ran_at), 'cid'));
    const again = el('button', 'Run again', 'linkbtn'); again.type = 'button'; again.addEventListener('click', () => run(e.id, [r.id])); bar.append(again);
  } else bar.append(el('span', c.state === 'not_run' ? 'Not run yet' : 'Running…', 'badge busy'));
  d.append(bar);
  if (e.scope) d.append(el('div', e.scope, 'dlabel'));
  if (c.error) d.append(el('div', 'Last run failed: ' + c.error, 'evline why'));
  if (c.state === 'done') {
    if (!c.items.length) d.append(el('div', e.empty || 'This session never called the scout agent, so there is nothing to check.', 'muted'));
    c.items.forEach(it => {
      const card = el('div', undefined, 'evcall ' + it.verdict);
      card.append(el('div', it.heading + ' · ' + (LABEL[it.verdict] || it.verdict), 'evh'));
      card.append(el('div', it.label, 'dlabel'), el('div', it.text, 'bubble big'));
      it.lines.forEach(t => card.append(el('div', t, 'evline')));
      (it.why || []).forEach(t => card.append(el('div', t, 'evline why')));
      if (it.memory) { const m = el('details', undefined, 'mem'); m.append(el('summary', it.memory_label || 'What the coordinator remembered at that moment'), el('pre', it.memory)); card.append(m); }
      d.append(card);
    });
  }
  d.hidden = false; $('#scrim').hidden = false;
  if (wasHidden) x.focus();
}
document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && openCell) closeDrawer(); });
$('#scrim').addEventListener('click', closeDrawer);

// The run number is the one on the All runs page: runs (test runs and app chats) newest first, the oldest is 1.
async function runNumbers() {
  let runs = [];
  try { runs = await (await fetch('/api/runs')).json(); } catch (e) { runs = []; }
  try { runs = runs.concat(await (await fetch('/api/app-runs')).json()).sort((x, y) => new Date(y.at) - new Date(x.at)); } catch (e) { /* the app is offline: test runs only */ }
  runNo = {}; runs.forEach((r, i) => { if (r.chat_id) runNo[r.chat_id] = runs.length - i; });
}
async function load() {
  const [d] = await Promise.all([fetch('/api/eval/sessions').then(r => r.json()), runNumbers()]);
  evals = d.evals; rows = d.rows; draw();
  // while anything is waiting or running, look again every few seconds
  clearTimeout(timer);
  if (rows.some(r => evals.some(e => ['queued', 'running'].includes(r.cells[e.id].state)))) timer = setTimeout(load, 3000);
}
$('#search').addEventListener('input', e => { query = e.target.value; draw(); });
filters(); load();
})();
