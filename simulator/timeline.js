// Turns a saved test chat into a timeline: who talked to whom, what they were given, what they returned.
(() => {
const chatId = location.pathname.split('/').pop();
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };

const YOU = ['You', 'you'], COORD = ['Coordinator', 'coord'], SCOUT = ['Scout agent', 'scout'];
const tool = name => [name, 'tool'];
const isScoutTool = n => n === 'scout_agent' || n === 'parser_agent';  // old saved chats used the earlier name

const show = v => v === null || v === undefined ? 'nothing' : typeof v === 'string' ? v : JSON.stringify(v);
const json = v => { try { return JSON.parse(v); } catch (e) { return v; } };

// What the scout agent returned, in words a person can read.
function report(d) {
  if (!d || typeof d !== 'object') return [['Returns', show(d)]];
  const rows = [['Status', d.status]];
  const found = ['business_type', 'area', 'city'].filter(k => d[k]).map(k => `${k.replace('_', ' ')}: ${d[k]}`);
  rows.push(['Found', found.length ? found.join(' · ') : 'nothing yet']);
  if (d.missing && d.missing.length) rows.push(['Missing', d.missing.join(', '), 'miss']);
  if (d.question) rows.push(['Question', d.question]);
  if (d.args) rows.push(['Search', `${d.args.query}  (lat ${d.args.lat}, lng ${d.args.lng})`]);
  if (d.problem) rows.push(['Problem', d.problem]);
  return rows;
}

function build(log) {
  const turns = []; let turn = null, answered = false;
  const start = (kind, text) => { turn = { n: turns.length + 1, kind, text, steps: [] }; turns.push(turn); };
  const add = (from, to, what, rows, cls) => turn && turn.steps.push({ from, to, what, rows, cls });
  let sawReport = false;
  for (const item of log) {
    if (item.u !== undefined) { const kind = answered ? 'answer' : 'message'; start(kind, item.u); answered = false; add(YOU, COORD, kind === 'answer' ? 'answers' : 'sends', [['Says', item.u]]); sawReport = false; continue; }
    const ev = item.ev; if (!ev) continue;
    if (!turn) start('message', '');
    if (ev.type === 'thought') add(COORD, COORD, 'says before acting', [['Says', ev.text]]);
    else if (ev.type === 'tool_call' && isScoutTool(ev.name)) add(COORD, SCOUT, 'hands over the request', [['Gives', (ev.args || {}).request], ['Why', 'Only the scout agent decides what is missing']]);
    else if (ev.type === 'tool_call' && ev.name === 'ask_user') add(COORD, tool('ask_user'), 'relays the question', [['Gives', (ev.args || {}).question]], 'ask');
    else if (ev.type === 'sub' && ev.kind === 'tool_call') add(SCOUT, tool(ev.name), 'looks up', [['Gives', show(ev.args)]]);
    else if (ev.type === 'sub' && ev.kind === 'tool_result') add(tool(ev.name), SCOUT, 'returns', [['Returns', show(json(ev.content))]]);
    else if (ev.type === 'sub' && ev.kind === 'result') { sawReport = true; add(SCOUT, COORD, 'reports back', report(ev.data)); }
    else if (ev.type === 'tool_result' && isScoutTool(ev.name)) { if (!sawReport) add(SCOUT, COORD, 'reports back', report(ev.data)); }
    else if (ev.type === 'tool_result' && ev.name === 'ask_user') add(tool('ask_user'), COORD, 'returns your answer', [['Returns', show(ev.data)]]);
    else if (ev.type === 'interrupt') { add(COORD, YOU, 'asks and waits', [['Asks', ev.question]], 'ask'); answered = true; }
    else if (ev.type === 'message') add(COORD, YOU, 'replies', [['Says', ev.text]]);
    else if (ev.type === 'error') add(['Error', 'tool'], COORD, 'failed', [['Message', ev.message || show(ev)]]);
  }
  return turns;
}

function render(chat) {
  $('#title').textContent = chat.title || 'Timeline';
  $('#chatlink').href = '/chat/' + chatId;
  const turns = build(chat.log || []), box = $('#turns');
  $('#empty').hidden = turns.length > 0;
  if (!turns.length) $('#empty').textContent = 'This chat has no recorded steps.';
  turns.forEach(t => {
    const card = el('section', undefined, 'turn'), h = el('h2');
    h.append(el('small', 'TURN ' + t.n), `${t.kind === 'answer' ? 'You answered' : 'You asked'}: ${t.text}`);
    card.append(h);
    t.steps.forEach((s, i) => {
      const row = el('div', undefined, 'step ' + (s.cls || '')), route = el('div', undefined, 'route'), io = el('dl', undefined, 'io');
      route.append(el('span', s.from[0], 'who ' + s.from[1]), el('span', '→', 'arrow'), el('span', s.to[0], 'who ' + s.to[1]), el('span', s.what, 'what'));
      (s.rows || []).forEach(([k, v, cls]) => {
        const dd = el('dd', String(v), (k === 'Returns' || k === 'Gives') && String(v).startsWith('{') ? 'mono' : cls || '');
        io.append(el('dt', k), dd);
      });
      const body = el('div'); body.append(route, io);
      row.append(el('span', `${t.n}.${i + 1}`, 'no'), body); card.append(row);
    });
    box.append(card);
  });
}

const get = path => fetch(path + chatId).then(r => r.ok ? r.json() : Promise.reject());
get('/api/chats/').catch(() => get('/api/app-chats/')).then(render)  // a test run, else a chat typed in the app
  .catch(() => { $('#empty').hidden = false; $('#empty').textContent = 'This chat is no longer in the simulator history (it was deleted).'; });
})();
