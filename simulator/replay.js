// Opens a saved test chat in the app's own page, read-only, with the run's result in a small card.
(() => {
const chatId = location.pathname.split('/').pop();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };

const style = el('style'); style.textContent = `
.sim{position:fixed;left:16px;bottom:14px;z-index:50;width:min(340px,calc(100vw - 32px));font:13px/1.45 var(--sans);color:var(--text);background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 40px -18px rgba(51,65,85,.55);overflow:hidden}
.sim-head{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line)}
.sim-head b{font:600 12px var(--mono);letter-spacing:.8px;color:var(--brand-deep)}
.sim-head{cursor:pointer;user-select:none}.sim-head:focus-visible{outline:3px solid var(--mint-bright);outline-offset:-3px}
.sim-head span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink);font-weight:500}
.sim-head .sim-res{flex:none;font:500 11px var(--mono);color:var(--muted)}.sim.fail .sim-res{color:var(--bad)}.sim.error .sim-res{color:var(--warn)}.sim.pass .sim-res{color:var(--brand-deep)}
.sim-chev{flex:none;color:var(--muted);transition:transform .15s}.sim.min .sim-chev{transform:rotate(-90deg)}
.sim.min .sim-head{border-bottom:0}.sim.min .sim-body{display:none}
.sim-dot{width:9px;height:9px;border-radius:50%;background:var(--brand)}.sim.fail .sim-dot{background:var(--bad)}.sim.error .sim-dot{background:var(--warn)}
.sim-body{padding:10px 12px 12px;display:flex;flex-direction:column;gap:8px}
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
const panel = el('div', undefined, 'sim min'), head = el('div', undefined, 'sim-head'), title = el('span', 'Saved run'), result = el('em', undefined, 'sim-res'), chev = el('i', '▾', 'sim-chev');
result.style.fontStyle = 'normal';
head.setAttribute('role', 'button'); head.tabIndex = 0; head.setAttribute('aria-expanded', 'false'); head.title = 'Show or hide the run result';
head.append(el('i', undefined, 'sim-dot'), el('b', 'SAVED RUN'), title, result, chev);
// a dropdown: closed, the card is one line and the chat behind it can be read; open, it shows the checks
const toggle = () => { panel.classList.toggle('min'); head.setAttribute('aria-expanded', String(!panel.classList.contains('min'))); };
head.addEventListener('click', toggle);
head.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
const body = el('div', undefined, 'sim-body'), stepEl = el('div', 'Opening the saved chat…', 'sim-step'), btns = el('div', undefined, 'sim-btns'), checksEl = el('ul', undefined, 'sim-checks');
const toRuns = el('a', 'All runs'); toRuns.href = '/runs'; const toDash = el('a', 'Test cases'); toDash.href = '/'; const toTimeline = el('a', 'Timeline'); toTimeline.href = '/timeline/' + chatId;
const checksBtn = el('button', 'Show checks'); checksBtn.hidden = true;
checksBtn.addEventListener('click', () => { checksEl.hidden = !checksEl.hidden; checksBtn.textContent = checksEl.hidden ? 'Show checks' : 'Hide checks'; });
btns.append(toRuns, toTimeline, toDash, checksBtn); body.append(stepEl, btns, checksEl); panel.append(head, body);

async function summary() {
  try {
    const find = async url => (await (await fetch(url)).json()).find(r => r.chat_id === chatId);
    const run = (await find('/api/runs')) || (await find('/api/app-runs'));
    if (!run) return;
    const kind = run.passed ? 'pass' : run.outcome === 'error' ? 'error' : 'fail', held = (run.checks || []).filter(k => k.ok).length;
    title.textContent = run.title; panel.classList.add(kind);
    result.textContent = `${run.passed ? 'Passed' : run.outcome === 'error' ? 'Errored' : 'Failed'} ${held}/${(run.checks || []).length}`;
    stepEl.textContent = `${run.passed ? 'Passed' : run.outcome === 'error' ? 'Errored' : 'Failed'} · ${held} of ${(run.checks || []).length} checks held${run.seconds != null ? ' · ' + run.seconds + ' s' : ''}. Read-only copy of what happened.`;
    (run.checks || []).forEach(k => { const li = el('li', undefined, k.ok ? 'ok' : 'no'); li.append(el('i', k.ok ? '✓' : '✗')); const t = el('span', k.label); if (k.detail) { t.append(' '); t.append(el('small', '(' + k.detail + ')')); } li.append(t); checksEl.append(li); });
    checksEl.hidden = run.passed; checksBtn.hidden = false; checksBtn.textContent = run.passed ? 'Show checks' : 'Hide checks';
  } catch (e) { /* the chat still opens without the summary */ }
}

// The page's own history drawer knows how to replay a saved chat, so open it and pick this chat.
async function openChat() {
  $('#open-history').click();
  for (let i = 0; i < 40; i++) {
    const row = document.querySelector(`.hrow .open[title*="${chatId}"]`);
    if (row) { row.click(); await sleep(300); return true; }
    await sleep(150);
  }
  $('#close-history').click(); return false;
}

document.addEventListener('DOMContentLoaded', async () => {
  document.head.append(style); document.body.append(panel);
  $('#composer').inert = true; $('#composer').style.opacity = '.45';  // a saved run cannot be continued
  summary();
  if (!(await openChat())) stepEl.textContent = 'This chat is no longer in the simulator history (it was deleted).';
  $('#composer').inert = true;
});
})();
