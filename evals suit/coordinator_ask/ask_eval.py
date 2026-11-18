"""Eval 2 (coordinator -> person): is every message the coordinator sends the person helpful for what they asked?

What counts as a message to the person: the question the coordinator asks with ask_user, and its final reply (an AI message with no tool call).
What never counts: the model's reasoning (reasoning_content), the note a model writes before a tool call, and the "Coordinator is thinking"
panel of the chat page (that panel is drawn by the page and is not in Langfuse at all).

Each message is judged on its own, with the conversation as it was when the coordinator wrote it (the person's messages and answers, the
coordinator's earlier messages, and what the scout agent reported, including the search results). It is NOT helpful when it:
  - off_topic    is about a different type of place, area or city than the person asked for (or about something unrelated), without saying why
  - asks_known   asks the person about something they already said (the area, the city, the type of place, or what they want)
  - no_results   comes after the search ran (or failed) but only says it is "prepared" instead of telling the person what was found or what went wrong
  - unhelpful    does nothing to move the person's request forward (empty, only an apology, a repeat)
It IS helpful when it asks a question about the same request (a preference, a cuisine, a budget), explains a wider search area, relays the
question the scout agent needs answered, answers with the search results, or tells a person who did not ask for a place what it can do.
A session PASSES only if every message passes.

The yes/no answers come from the Gemini Decisions API (decide.py): one request per message, a few questions about it.

usage:
  python ask_eval.py <session_id> [<session_id> ...]
  python ask_eval.py --all            every session in Langfuse that has a coordinator message
  add --show to print the conversation of each result

Files here: ask_results.json (one result per message), ask_sessions.json (when each session was evaluated). Nothing is cached.
"""
import collections, json, re, sys, threading, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'scout_fields'))
import eval_core  # Langfuse access
from decide import decide

LOCK = threading.Lock()
SURE = 0.70  # below this a Decisions answer is not trusted: the message goes to "review"

# Yes/no questions asked about one message. "yes" to any of them is a fail. The text sent with them has a CONVERSATION and then the COORDINATOR message.
QUESTIONS = {
    'off_topic': 'Is the COORDINATOR message about a different type of place, a different area or a different city than the person asked for, or about something that has '
                 'nothing to do with their request, without explaining why? Answer no for: a question about a preference for the same request (a cuisine, a style, '
                 'a budget, a time); a wider or nearby search area that the message explains; an answer, or an honest "I cannot do that", about what the person asked; '
                 'a short reply that says what the assistant can do when the person did not ask for a place.',
    'asks_known': 'Did the person already tell, in the CONVERSATION (read their last message first), the very thing that the COORDINATOR message asks them about (for example the area, '
                  'the city, the type of place, or what they want)? Asking for another spelling, or for a nearby landmark, of a place that could not be found is not asking about something already told: answer no for that.',
    'unhelpful': 'Is the COORDINATOR message of no help to the person: it is empty, only an apology, repeats what was said before, or does nothing to move their request '
                 'forward? A question that gets a detail the search needs, a preference question about the request, a clear answer, a short note on what the '
                 'assistant can do, and a clear honest statement that it cannot do something or does not have that information are all helpful, so answer no for them.',
}
# Asked only for a reply that follows the scout agent's search outcome: the person must be told what was found (or what went wrong).
NO_RESULTS = ('no_results', "The CONVERSATION ends with the scout agent's search outcome. Does the COORDINATOR message only say that something is prepared, ready or "
                            'on its way, without telling the person what the search found (how many places or which places) or what went wrong?')

# Asked on a shorter text, only the person's last message and the coordinator message: in a long conversation the full-text answer can miss that
# the person said it just now. Its answer counts as `asks_known` (the higher of the two chances is used).
ASKS_LAST = ('asks_known', 'Would the person find the COORDINATOR question pointless, because their own message already answers it?')

NOTES = {
    'off_topic': 'The message is about something other than what the person asked for.',
    'asks_known': 'The coordinator asked for something the person had already said.',
    'no_results': 'The search had results (or a problem), but the reply did not tell the person what they were.',
    'unhelpful': 'The message does not help the person with what they asked.',
}


def _read(name, default):
    try:
        return json.loads((HERE / name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _write(name, data):
    tmp = HERE / (name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(HERE / name)


def _tool_names(messages):
    """The names of the tools a model call was given (the tool definitions sit in its input as messages)."""
    names = set()
    for m in messages if isinstance(messages, list) else []:
        content = m.get('content') if isinstance(m, dict) else None
        if isinstance(content, dict) and isinstance(content.get('function'), dict):
            names.add(content['function'].get('name'))
    return names


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# ---- the conversation, as the coordinator saw it ----
def scout_text(report):
    """What the scout agent reported, in one line: its status, the details, the question it wants asked, what the search found, the problem."""
    if not isinstance(report, dict):
        return _text(report)[:200]
    parts = [f'status {report.get("status")}']
    for key, label in (('business_type', 'business'), ('area', 'area'), ('city', 'city')):
        if report.get(key):
            parts.append(f'{label} {report[key]}')
    if report.get('missing'):
        parts.append('missing ' + ' and '.join(report['missing']))
    if report.get('question'):
        parts.append(f'it wants the person asked "{report["question"]}"')
    found = report.get('search') or {}
    if found.get('count'):
        best = (found.get('highlights') or {}).get('best') or {}
        parts.append(f'search found {found["count"]} places' + (f'; best overall {best["name"]} ({best["rating"]} stars, {best["reviews"]} reviews)' if best else ''))
    if report.get('problem'):
        parts.append(f'problem: {report["problem"]}')
    return '; '.join(parts)


def render(messages):
    """The coordinator's memory as plain lines: the person, the coordinator, the scout agent's reports. The system prompt and tool definitions are left out."""
    lines = []
    for m in messages if isinstance(messages, list) else []:
        role, content = m.get('role'), m.get('content')
        if role == 'system' or (role == 'tool' and isinstance(content, dict) and 'function' in content):  # the system prompt, the tool definitions
            continue
        if role == 'user':
            lines.append(f'Person: {_text(content)}')
        elif role == 'assistant':
            for call in m.get('tool_calls') or []:
                args = call.get('args') or {}
                if call['name'] == 'scout_agent':
                    lines.append(f'Coordinator asked the scout agent: "{args.get("request")}"')
                elif call['name'] == 'ask_user':
                    lines.append(f'Coordinator asked the person: "{args.get("question")}"')
            if _text(content).strip() and not m.get('tool_calls'):
                lines.append(f'Coordinator said: {_text(content).strip()}')
        elif role == 'tool':
            if m.get('name') == 'ask_user':
                lines.append(f'Person answered: {_text(content)}')
            elif m.get('name') == 'scout_agent':
                lines.append('Scout agent reported: ' + scout_text(eval_core._parse(content)))
    return '\n'.join(lines)


def outcome(messages):
    """'results' or 'problem' when the last thing the scout agent reported is a finished search or a failure; otherwise None."""
    last = None
    for m in messages if isinstance(messages, list) else []:
        if m.get('role') == 'tool' and m.get('name') == 'scout_agent':
            last = eval_core._parse(m.get('content'))
    if not isinstance(last, dict):
        return None
    if last.get('status') == 'failed':
        return 'problem'
    return 'results' if last.get('status') == 'ready' and (last.get('search') or {}).get('count') else None


# ---- the messages of a session ----
def build_events(gens):
    """Oldest first: {'kind': 'ask' | 'reply', 'time', 'text', 'memory', 'outcome'}. Only the coordinator's own model calls are read (the ones given the
    scout_agent or ask_user tool), and only the question of an ask_user call or the text of a call that made no tool call. The reasoning and the text
    written beside a tool call are never read. `memory` is the conversation as that model call saw it."""
    events = []
    for o in sorted(gens, key=lambda o: o['startTime']):
        out, inp = eval_core._parse(o.get('output')), eval_core._parse(o.get('input'))
        if not _tool_names(inp) & {'scout_agent', 'ask_user'}:  # the scout agent's own model calls, or anything else
            continue
        calls = out.get('tool_calls') if isinstance(out, dict) else None
        content = out.get('content') if isinstance(out, dict) else out
        base = {'time': o['startTime'], 'memory': render(inp), 'outcome': outcome(inp)}
        for call in calls or []:
            if call.get('name') == 'ask_user':
                events.append({**base, 'kind': 'ask', 'text': _text((call.get('args') or {}).get('question'))})
        if not calls and isinstance(content, str) and content.strip():
            events.append({**base, 'kind': 'reply', 'text': content.strip()})
    return events


def pull_events(session_id):
    flt = [{'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': session_id}, {'type': 'string', 'column': 'type', 'operator': '=', 'value': 'GENERATION'}]
    return build_events(eval_core._paged(flt))


def pull_all():
    """session id -> events, for every session in Langfuse."""
    by = collections.defaultdict(list)
    for o in eval_core._paged([{'type': 'string', 'column': 'type', 'operator': '=', 'value': 'GENERATION'}]):
        by[o.get('sessionId')].append(o)
    return {sid: build_events(g) for sid, g in by.items() if sid}


# ---- judging one message ----
_MARKS = ('Person: ', 'Person answered: ', 'Coordinator ', 'Scout agent ')


def last_person(memory):
    """The person's last message or answer in the conversation text (it can span several lines), or ''."""
    out, take = [], False
    for line in (memory or '').split('\n'):
        if line.startswith(_MARKS):
            take = line.startswith(('Person: ', 'Person answered: '))
            if take:
                out = [line.split(': ', 1)[1]]
        elif take:
            out.append(line)
    return '\n'.join(out).strip()


def _tidy(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', (text or '').lower())).strip()


def relays_scout(memory, message):
    """True when the message is, word for word, the question the scout agent's last report asked the coordinator to put to the person."""
    asked = [l for l in (memory or '').split('\n') if l.startswith('Scout agent reported:') and 'it wants the person asked "' in l]
    got = re.search(r'it wants the person asked "(.*?)"(?:;|$)', asked[-1]) if asked else None
    return bool(got) and _tidy(got.group(1)) == _tidy(message)


def judge(memory, kind, message, found=None):
    """-> {verdict, failures, notes, scores}. verdict: pass | fail | review. `found`: 'results' / 'problem' when the scout agent's search outcome is in the conversation."""
    questions = dict(QUESTIONS)
    if kind == 'reply' and found:
        questions[NO_RESULTS[0]] = NO_RESULTS[1]
    channel = 'a question the coordinator asked the person' if kind == 'ask' else "the coordinator's reply to the person"
    text = f'CONVERSATION:\n{memory or "(empty)"}\n\nCOORDINATOR message to check ({channel}):\n"{message}"'
    got = decide(text, questions)
    last = last_person(memory) if kind == 'ask' and not relays_scout(memory, message) else ''  # a question the scout agent asked for is not the coordinator's own
    if last:  # a question can repeat what the person said a moment ago
        short = decide(f'PERSON message:\n"{last}"\n\nCOORDINATOR message (a question the coordinator asked the person):\n"{message}"', {ASKS_LAST[0]: ASKS_LAST[1]})
        if short[ASKS_LAST[0]]['p_yes'] > got[ASKS_LAST[0]]['p_yes']:
            got[ASKS_LAST[0]] = short[ASKS_LAST[0]]
    failures, unsure = [], []
    for name, a in got.items():
        if a['confidence'] < SURE:  # an answer the model is not sure of is never counted as a fail or a pass
            unsure.append(name)
        elif a['yes']:
            failures.append(name)
    verdict = 'fail' if failures else 'review' if unsure else 'pass'
    notes = [NOTES[f] for f in failures]
    if verdict == 'review':
        notes.append('The Decisions API was not sure (confidence under %d%%) on: %s.' % (SURE * 100, ', '.join(sorted(unsure))))
    return {'verdict': verdict, 'failures': failures, 'notes': notes, 'scores': {k: round(v['p_yes'], 3) for k, v in got.items()}}


def evaluate(session_id, events):
    return [{'item_id': f'{session_id}:{n}', 'session_id': session_id, 'time': e['time'], 'channel': e['kind'], 'message': e['text'], 'memory': e['memory'],
             'outcome': e['outcome'], **judge(e['memory'], e['kind'], e['text'], e['outcome'])} for n, e in enumerate(events, 1)]


def session_verdict(results):
    if not results:
        return 'no messages'
    verdicts = {r['verdict'] for r in results}
    return 'fail' if 'fail' in verdicts else 'review' if 'review' in verdicts else 'pass'


def save(session_id, results):
    with LOCK:
        old = [r for r in _read('ask_results.json', []) if r.get('session_id') != session_id]  # a session evaluated again replaces its earlier results
        _write('ask_results.json', sorted(old + results, key=lambda r: r['time']))
        runs = _read('ask_sessions.json', {})
        runs[session_id] = {'ran_at': time.time(), 'items': len(results), 'verdict': session_verdict(results), 'model': 'gemini-decision-1'}
        _write('ask_sessions.json', runs)


def run_session(session_id):
    results = evaluate(session_id, pull_events(session_id))
    save(session_id, results)
    return results


def run(session_ids, show=False):
    for sid in session_ids:
        results = run_session(sid)
        print(f'\n{session_verdict(results).upper():12} session {sid}  ({len(results)} checked)')
        for r in results:
            print(f'   {r["verdict"].upper():7} {",".join(r["failures"]):26} coordinator[{r["channel"]}]: {r["message"]!r}')
            print('           chance of yes: ' + ' '.join(f'{k} {v}' for k, v in r['scores'].items()))
            if show:
                print('           conversation:\n              ' + r['memory'].replace('\n', '\n              '))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--all' in sys.argv:
        args = sorted(pull_all())
    if not args:
        sys.exit(__doc__)
    run(args, show='--show' in sys.argv)
