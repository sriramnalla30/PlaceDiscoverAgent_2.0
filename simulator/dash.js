(() => {
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const NAMES = { business_type: 'the type of place', area: 'the area', city: 'the city' };
const list = a => a.map(x => NAMES[x] || x).join(' and ');

function expectation(c) {
  const e = c.expect;
  if (!e.firstMissing) return 'Scout agent says ready on the first message. The coordinator asks nothing and prints the search arguments.';
  const ask = 'Scout agent says it is missing ' + list(e.firstMissing) + ', so the coordinator asks';
  return ask + (e.questionMentions ? ' for another spelling or a nearby landmark' : '') + (e.asks > 1 ? '. The first answer is not enough, so it asks again, and says ready after the second.' : '. After the answer it says ready.');
}
function ago(t) {
  const m = Math.round((Date.now() - new Date(t).getTime()) / 60000);
  return m < 1 ? 'just now' : m < 60 ? m + ' min ago' : m < 1440 ? Math.round(m / 60) + ' h ago' : new Date(t).toLocaleDateString();
}

function card(c) {
  const box = el('article', undefined, 'card'), top = el('div', undefined, 'top');
  const last = c.last, badge = el('span', last ? (last.passed ? 'Passed' : last.outcome === 'error' ? 'Errored' : 'Failed') : 'Not run yet', 'badge' + (last ? (last.passed ? ' pass' : last.outcome === 'error' ? ' error' : ' fail') : ''));
  top.append(el('span', c.id.toUpperCase(), 'tag'), badge);
  const say = el('div', undefined, 'say'); say.append(el('small', 'The fake user types'));
  say.append(el('div', c.first, 'bubble'));
  c.replies.forEach(r => { say.append(el('small', 'then answers')); say.append(el('div', r, 'bubble reply')); });
  const exp = el('div', undefined, 'expect'); exp.append(el('small', 'Expected'), el('p', expectation(c)));
  if (c.target) exp.append(el('small', 'Hunts this failure'), el('p', c.target));
  box.append(top, el('h3', c.title), el('p', c.persona, 'persona'), say, exp);
  if (last) {
    const l = el('div', undefined, 'last'); l.append(el('span', `Last run ${ago(last.at)}${last.seconds ? ' · ' + last.seconds + ' s' : ''}`));
    const bad = (last.checks || []).filter(k => !k.ok);
    if (bad.length) { const ul = el('ul'); bad.forEach(k => ul.append(el('li', k.label + (k.detail ? ': ' + k.detail : '')))); l.append(ul); }
    box.append(l);
  }
  const act = el('div', undefined, 'actions'), run = el('a', 'Run ', 'run'); run.href = '/run/' + c.id; run.append(el('span', '↗'));
  run.setAttribute('aria-label', 'Run test case ' + c.title); act.append(run); box.append(act);
  return box;
}

async function load() {
  try {
    const cases = await (await fetch('/api/cases')).json();
    ['happy', 'ambiguity', 'edge'].forEach(g => {
      const holder = $('#' + g + ' .cards'), group = cases.filter(c => c.group === g); holder.replaceChildren(); group.forEach(c => holder.append(card(c)));
      const head = $('#' + g + ' .ghead'); head.querySelectorAll('.runall').forEach(x => x.remove());
      if (group.length) {  // one click runs the group's cases one after another in the run page
        const all = el('a', `Run all ${group.length} `, 'runall'); all.append(el('span', '↗'));
        all.href = `/run/${group[0].id}?queue=${group.slice(1).map(c => c.id).join(',')}&n=1&of=${group.length}`;
        all.setAttribute('aria-label', `Run all ${group.length} ${g === 'happy' ? 'happy flow' : g === 'edge' ? 'edge case' : 'ambiguity'} test cases, one after another`); head.append(all);
      }
    });
  } catch (e) { $('#happy .cards').textContent = 'Could not load the test cases.'; }
  try {
    const s = await (await fetch('/api/app-status')).json(), box = $('#status');
    box.className = 'status ' + (s.online ? 'on' : 'off');
    box.querySelector('span').textContent = s.online ? `App under test online · ${s.url}${s.model ? ' · ' + s.model : ''}` : `App under test not reachable at ${s.url}. Start it before running a case.`;
  } catch (e) { /* the status chip keeps its first text */ }
}
load();
window.addEventListener('pageshow', e => { if (e.persisted) load(); });
})();
