"""Eval 3 (coordinator -> scout hand-off): did the coordinator write the right arguments for each scout_agent call?

For every scout_agent call in a Langfuse session:
  - the memory the coordinator had at that moment is the input of the coordinator model call that made the call
  - one LLM-judge call reads that memory and the request the coordinator wrote, and answers Pass or Fail
A session PASSES only if every one of its scout calls passes. One failing call fails the session.

usage:
  python handoff_eval.py <session_id> [<session_id> ...]
  python handoff_eval.py --all            every session that has a scout call (Langfuse)
  python handoff_eval.py <session_id> --show   print each call's memory and critique

Files here: handoff_results.json (one result per call), handoff_sessions.json (one verdict per session). Nothing is cached: every run asks the judge again.
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'scout_fields'))
import eval_core  # Langfuse access and the judge model, shared with eval 1

_prompt = None


def judge_prompt():
    """The judge prompt from Langfuse prompt management (name handoff-judge, label draft). Save the file there with sync_prompt.py."""
    global _prompt
    if _prompt is None:
        from langfuse import Langfuse
        client = Langfuse(public_key=eval_core.ENV['LANGFUSE_PUBLIC_KEY'], secret_key=eval_core.ENV['LANGFUSE_SECRET_KEY'], base_url=eval_core.LANGFUSE)
        _prompt = client.get_prompt('handoff-judge', type='text', label=eval_core.PROMPT_LABEL, cache_ttl_seconds=0)
    return _prompt


FIELDS = 'core,basic,io'
WINDOW = {'fromStartTime': '2026-10-01T00:00:00Z', 'toStartTime': '2027-12-31T00:00:00Z'}


def _read(name, default):
    try:
        return json.loads((HERE / name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _write(name, data):
    tmp = HERE / (name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(HERE / name)


def _text(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def render_memory(messages) -> str:
    """The coordinator's saved conversation as plain lines. The system prompt and the tool definitions are left out."""
    lines = []
    for m in messages:
        role, content = m.get('role'), m.get('content')
        if role == 'system' or (role == 'tool' and isinstance(content, dict)):  # system prompt, tool definitions
            continue
        if role == 'user':
            lines.append(f'Person: {_text(content)}')
        elif role == 'assistant':
            for call in m.get('tool_calls') or []:
                args = call.get('args') or {}
                if call['name'] == 'scout_agent':
                    lines.append(f'Coordinator called scout_agent(request="{args.get("request")}")')
                elif call['name'] == 'ask_user':
                    lines.append(f'Coordinator asked the person: "{args.get("question")}"')
                else:
                    lines.append(f'Coordinator called {call["name"]}({json.dumps(args, ensure_ascii=False)})')
            if _text(content).strip() and not m.get('tool_calls'):
                lines.append(f'Coordinator said: {_text(content).strip()}')
        elif role == 'tool':
            if m.get('name') == 'ask_user':
                lines.append(f'Person answered: {_text(content)}')
            elif m.get('name') == 'scout_agent':
                report = eval_core._parse(content)
                if isinstance(report, dict):
                    miss = report.get('missing') or []
                    lines.append(f'scout_agent returned: {report.get("status")}' + (f', missing {" and ".join(miss)}' if miss else ''))
                else:
                    lines.append(f'scout_agent returned: {_text(content)[:200]}')
    return '\n'.join(lines)


def pull_calls(session_id=None):
    """Every coordinator model call that decided to call scout_agent, oldest first, with the memory it had."""
    flt = [{'type': 'string', 'column': 'name', 'operator': '=', 'value': 'Coordinator: model call'}]
    if session_id:
        flt.append({'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': session_id})
    rows, cursor = [], None
    while True:
        q = {'filter': json.dumps(flt), **WINDOW, 'limit': 100, 'fields': FIELDS}
        if cursor:
            q['cursor'] = cursor
        page = eval_core._observations(q)
        rows += page['data']
        cursor = (page.get('meta') or {}).get('cursor')
        if not cursor:
            break
    calls = []
    for o in sorted(rows, key=lambda o: o['startTime']):
        out = eval_core._parse(o.get('output'))
        scout = [c for c in (out.get('tool_calls') or []) if c.get('name') == 'scout_agent'] if isinstance(out, dict) else []
        inp = eval_core._parse(o.get('input'))
        if not scout or not isinstance(inp, list):
            continue
        for k, call in enumerate(scout):
            req = (call.get('args') or {}).get('request')
            calls.append({'call_id': f'{o["id"]}:{k}', 'session_id': o.get('sessionId'), 'time': o['startTime'],
                          'request': req if isinstance(req, str) else json.dumps(req, ensure_ascii=False), 'memory': render_memory(inp)})
    return calls


def judge(call):
    """One LLM call for one scout call, asked fresh every time."""
    prompt = judge_prompt()
    prompt_id = f'v{prompt.version}'
    read = eval_core._ask(prompt.compile(memory=call['memory'], request=call['request']))
    verdict = str((read or {}).get('result', '')).strip().lower()
    return {**call, 'verdict': verdict if verdict in ('pass', 'fail') else 'unknown',
            'failure_type': (read or {}).get('failure_type'), 'critique': (read or {}).get('critique'),
            'judge_model': eval_core.MODEL, 'prompt': prompt_id}


def session_verdict(results):
    """pass only when every call passed; one fail fails the session; a call the judge could not answer makes it unknown."""
    if not results:
        return 'no scout call'
    if any(r['verdict'] == 'fail' for r in results):
        return 'fail'
    return 'pass' if all(r['verdict'] == 'pass' for r in results) else 'unknown'


def run_session(session_id):
    """Judge every scout call of one session and save the results. Returns the per-call results."""
    results = [judge(c) for c in pull_calls(session_id)]
    merged = {r['call_id']: r for r in _read('handoff_results.json', [])}
    merged.update({r['call_id']: r for r in results})
    _write('handoff_results.json', sorted(merged.values(), key=lambda r: r['time']))
    sessions = _read('handoff_sessions.json', {})
    sessions[session_id] = {'verdict': session_verdict(results), 'calls': len(results), 'failed_calls': sum(r['verdict'] == 'fail' for r in results),
                            'ran_at': time.time(), 'judge_model': eval_core.MODEL, 'prompt': f'v{judge_prompt().version}'}
    _write('handoff_sessions.json', sessions)
    return results


def run(session_ids, show=False):
    summary = {}
    for sid in session_ids:
        results = run_session(sid)
        verdict = session_verdict(results)
        summary[sid] = verdict
        print(f'\n{verdict.upper():14} session {sid}  ({len(results)} scout call(s))')
        for i, r in enumerate(results, 1):
            print(f'   call {i}: {r["verdict"].upper():7} {r["failure_type"] or "":9} request={r["request"]!r}')
            if show or r['verdict'] != 'pass':
                print('      memory:\n         ' + r['memory'].replace('\n', '\n         '))
                print(f'      critique: {r["critique"]}')
    return summary


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--all' in sys.argv:
        args = sorted({c['session_id'] for c in pull_calls() if c['session_id']})
    if not args:
        sys.exit(__doc__)
    summary = run(args, show='--show' in sys.argv)
    print('\n', {v: sum(1 for s in summary.values() if s == v) for v in sorted(set(summary.values()))})
