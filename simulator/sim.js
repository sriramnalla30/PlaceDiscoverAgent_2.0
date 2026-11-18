// The fake user. It types into the real chat page, answers the app's questions from the case's script,
// and checks what the scout agent decided. It reads the app's event stream but never changes what the page shows.
(() => {
const caseId = location.pathname.split('/').pop();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const ABORT = new Error('stopped');
// "Run all" opens the first case with the rest in ?queue=; each case hands over to the next when it finishes.
const params = new URLSearchParams(location.search);
const queue = (params.get('queue') || '').split(',').filter(Boolean), pos = +params.get('n') || 1, total = +params.get('of') || 1;
let paused = false, aborted = false, threadId = '', events = [];

// Tap the /api/chat stream (a clone, so the page reads its own copy untouched).
const realFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const res = await realFetch(input, init);
  try {
    const url = typeof input === 'string' ? input : input.url;
    if (url.endsWith('/api/chat') && init && init.body && res.body) {
      threadId = JSON.parse(init.body).thread_id || threadId;
      tap(res.clone());
    }
  } catch (e) { /* tapping is best effort; the page keeps working */ }
  return res;
};
async function tap(resp) {
  const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = '';
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true }); const parts = buf.split('\n\n'); buf = parts.pop();
    parts.forEach(p => { const line = p.split('\n').find(l => l.startsWith('data: ')); if (line) { try { events.push(JSON.parse(line.slice(6))); } catch (e) { /* ignore */ } } });
  }
}

// ---- the small control card, bottom-left ----
const style = el('style'); style.textContent = `
.sim{position:fixed;left:16px;bottom:14px;z-index:50;width:min(340px,calc(100vw - 32px));font:13px/1.45 var(--sans);color:var(--text);background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 40px -18px rgba(51,65,85,.55);overflow:hidden}
.sim-head{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer;user-select:none}
.sim-head b{font:600 12px var(--mono);letter-spacing:.8px;color:var(--brand-deep)}
.sim-head span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink);font-weight:500}
.sim-head a{font:500 12px var(--sans);color:var(--muted);text-decoration:none}.sim-head a:hover{color:var(--brand-deep)}
.sim-dot{width:9px;height:9px;border-radius:50%;background:var(--brand);animation:simpulse 1.2s infinite}
.sim.done .sim-dot,.sim.stopped .sim-dot{animation:none}.sim.pass .sim-dot{background:var(--brand)}.sim.fail .sim-dot{background:var(--bad)}.sim.stopped .sim-dot{background:var(--faint)}.sim.error .sim-dot{background:var(--warn)}
@keyframes simpulse{50%{opacity:.35}}
.sim-body{padding:10px 12px 12px;display:flex;flex-direction:column;gap:8px}
.sim.min .sim-body{display:none}
.sim-chev{flex:none;font-style:normal;color:var(--muted);transition:transform .15s}.sim.min .sim-chev{transform:rotate(-90deg)}
.sim-step{color:var(--ink)}
.sim-btns{display:flex;gap:8px;flex-wrap:wrap}
.sim-btns button,.sim-btns a{font:500 12px var(--sans);color:var(--ink);background:#fff;border:1px solid var(--faint);border-radius:999px;padding:5px 12px;cursor:pointer;text-decoration:none}
.sim-btns button:hover,.sim-btns a:hover{background:var(--mint);border-color:var(--mint-line)}
.sim-checks{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:3px;font-size:12.5px}
.sim-checks[hidden]{display:none}
.sim-checks li{display:flex;gap:7px}.sim-checks li i{font-style:normal;font-weight:700}
.sim-checks li.ok i{color:var(--brand)}.sim-checks li.no{color:var(--bad)}.sim-checks li.no i{color:var(--bad)}
.sim-checks small{color:var(--muted)}
@media(max-width:900px){.sim{bottom:auto;top:70px}}
`;
const panel = el('div', undefined, 'sim');
const head = el('div', undefined, 'sim-head'), title = el('span', 'Loading…'), back = el('a', 'Dashboard'); back.href = '/';
const chev = el('i', '▾', 'sim-chev');
head.append(el('i', undefined, 'sim-dot'), el('b', 'SIMULATOR'), title, back, chev);
const body = el('div', undefined, 'sim-body'), stepEl = el('div', 'Starting…', 'sim-step'), btns = el('div', undefined, 'sim-btns'), checksEl = el('ul', undefined, 'sim-checks');
const pauseBtn = el('button', 'Pause'), stopBtn = el('button', 'Stop'), againBtn = el('button', 'Run again'); againBtn.hidden = true;
const toDash = el('a', 'Test cases'); toDash.href = '/'; toDash.hidden = true;
const toRuns = el('a', 'All runs'); toRuns.href = '/runs'; toRuns.hidden = true;
const checksBtn = el('button', 'Show checks'); checksBtn.hidden = true;
checksBtn.addEventListener('click', () => { checksEl.hidden = !checksEl.hidden; checksBtn.textContent = checksEl.hidden ? 'Show checks' : 'Hide checks'; });
btns.append(pauseBtn, stopBtn, againBtn, toRuns, toDash, checksBtn); body.append(stepEl, btns, checksEl); panel.append(head, body);
head.addEventListener('click', e => { if (e.target !== back) panel.classList.toggle('min'); });
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; if (paused) step('Paused'); });
stopBtn.addEventListener('click', () => { aborted = true; });
againBtn.addEventListener('click', () => location.reload());
function step(text) { stepEl.textContent = text; }
function finish(kind) { panel.classList.remove('pass', 'fail', 'error', 'stopped'); panel.classList.add('done', kind); pauseBtn.hidden = stopBtn.hidden = true; againBtn.hidden = toDash.hidden = toRuns.hidden = false; }

// ---- helpers that behave like a person: type letter by letter, pause to read ----
async function gate() { while (paused && !aborted) await sleep(150); if (aborted) throw ABORT; }
async function wait(ms) { await gate(); await sleep(ms); await gate(); }
async function until(test, ms, what) {
  const end = Date.now() + ms;
  for (;;) { await gate(); if (test()) return; if (Date.now() > end) throw new Error('Timed out waiting for ' + what); await sleep(200); }
}
async function typeAndSend(text) {
  const input = $('#input'); input.focus();
  for (const ch of text) { input.value += ch; input.dispatchEvent(new Event('input', { bubbles: true })); await wait(35 + Math.random() * 45); }
  await wait(450);
  $('#send').click();
}

// ---- running one case ----
async function run() {
  const cases = await (await realFetch('/api/cases')).json();
  const c = cases.find(x => x.id === caseId);
  if (!c) { title.textContent = 'Unknown test case'; step('This test case does not exist.'); finish('error'); return; }
  title.textContent = (total > 1 ? `[${pos}/${total}] ` : '') + c.title;
  const started = Date.now(), turns = [c.first, ...c.replies];
  let problem = null, unexpectedQuestion = null;
  step('Opening the app…'); await until(() => $('#input') && !$('#input').disabled, 15000, 'the chat box'); await wait(1500);
  for (let i = 0; i < turns.length; i++) {
    step(i === 0 ? 'Typing the first message…' : 'Reading the coordinator\'s question, then answering…');
    if (i > 0) await wait(1800);
    const seen = events.length;
    await typeAndSend(turns[i]);
    step('Waiting for the coordinator…');
    await until(() => $('#input').disabled, 5000, 'the coordinator to start');
    await until(() => !$('#input').disabled && events.slice(seen).some(e => e.type === 'done'), 180000, 'the coordinator to finish');
    const fresh = events.slice(seen), err = fresh.find(e => e.type === 'error');
    if (err) { problem = err.message; break; }
    const done = fresh.filter(e => e.type === 'done').pop();
    if (!(done && done.waiting)) break;                       // the coordinator finished: nothing more to answer
    if (i === turns.length - 1) { unexpectedQuestion = (fresh.filter(e => e.type === 'interrupt').pop() || {}).question; break; }
  }
  await wait(600);
  const seconds = Math.round((Date.now() - started) / 1000);
  if (problem) return report(c, { passed: false, outcome: 'error', checks: [{ ok: false, label: 'The coordinator answered without an error', detail: problem }], seconds });
  report(c, { ...evaluate(c, unexpectedQuestion), seconds });
}

function evaluate(c, unexpectedQuestion) {
  const e = c.expect, checks = [], add = (ok, label, detail) => checks.push({ ok, label, detail });
  const results = events.filter(x => x.type === 'tool_result' && x.name === 'scout_agent' && x.data && typeof x.data === 'object').map(x => x.data);
  const statuses = results.map(r => r.status), asks = events.filter(x => x.type === 'interrupt');
  const same = JSON.stringify(statuses) === JSON.stringify(e.statuses);
  add(same, 'Scout agent decisions were ' + e.statuses.join(' → '), same ? '' : 'got ' + (statuses.join(' → ') || 'none'));
  add(asks.length === e.asks, e.asks === 0 ? 'The coordinator asked nothing' : `The coordinator asked ${e.asks} question${e.asks > 1 ? 's' : ''}`, asks.length === e.asks ? '' : 'asked ' + asks.length);
  if (unexpectedQuestion) add(false, 'The coordinator asked nothing more after the answer', unexpectedQuestion);
  if (e.firstMissing) {
    const first = results.find(r => r.status === 'needs_input'), missing = (first && first.missing) || [];
    const ok = e.firstMissing.every(m => missing.includes(m));
    add(ok, 'First question was about ' + e.firstMissing.join(' and '), ok ? '' : 'missing was [' + missing.join(', ') + ']');
  }
  if (e.questionMentions) {
    const q = (asks[0] && asks[0].question) || '', ok = new RegExp(e.questionMentions, 'i').test(q);
    add(ok, 'The question asks for another spelling or a landmark', ok ? '' : q);
  }
  const last = results[results.length - 1] || {}, args = last.args || {};
  add(last.status === 'ready' && typeof args.lat === 'number' && typeof args.lng === 'number', 'Finished with a map pin (lat and lng)', last.status === 'ready' ? '' : 'final status ' + (last.status || 'none'));
  [['business', last.business_type], ['area', last.area], ['city', last.city]].forEach(([key, value]) => {
    if (!e[key]) return;
    const ok = String(value || '').toLowerCase().includes(e[key]);
    add(ok, `${key} is "${e[key]}"`, ok ? '' : 'got "' + (value || '') + '"');
  });
  return { passed: checks.every(k => k.ok), outcome: checks.every(k => k.ok) ? 'passed' : 'failed', checks };
}

async function advance() {
  const base = stepEl.textContent;
  if (!queue.length) { if (total > 1) step(`${base} That was case ${pos} of ${total}. All runs shows the whole batch.`); return; }
  const [nextId, ...rest] = queue; let cancelled = false;
  const cancel = el('button', 'Cancel queue'); btns.append(cancel); cancel.addEventListener('click', () => { cancelled = true; });
  for (let left = 5; left > 0 && !cancelled; left--) { step(`${base} Next case in ${left} s…`); await sleep(1000); }
  cancel.remove();
  if (cancelled) { step(`${base} Queue cancelled.`); return; }
  location.href = `/run/${nextId}?queue=${rest.join(',')}&n=${pos + 1}&of=${total}`;
}

async function report(c, result) {
  const held = result.checks.filter(k => k.ok).length;
  step(result.passed ? `Passed. All ${held} checks held.` : result.outcome === 'error' ? 'The coordinator hit an error. Run again.' : `Failed. ${held} of ${result.checks.length} checks held.`);
  checksEl.replaceChildren(); checksEl.hidden = result.passed;  // a pass stays out of the way of the page; a failure opens its reasons
  checksBtn.hidden = false; checksBtn.textContent = result.passed ? 'Show checks' : 'Hide checks';
  result.checks.forEach(k => { const li = el('li', undefined, k.ok ? 'ok' : 'no'); li.append(el('i', k.ok ? '✓' : '✗')); const t = el('span', k.label); if (k.detail) { t.append(' '); t.append(el('small', '(' + k.detail + ')')); } li.append(t); checksEl.append(li); });
  finish(result.passed ? 'pass' : result.outcome === 'error' ? 'error' : 'fail');
  try { await realFetch('/api/results/' + c.id, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...result, chat_id: threadId, at: new Date().toISOString() }) }); } catch (e) { /* the result still shows here */ }
  await advance();
}

document.addEventListener('DOMContentLoaded', () => {
  document.head.append(style); document.body.append(panel);
  run().catch(e => { if (e === ABORT) { step('Stopped. Nothing was saved.'); finish('stopped'); } else { step('The simulator stopped: ' + e.message); finish('error'); advance(); } });
});
})();
