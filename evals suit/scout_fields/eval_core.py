"""Eval 1 (scout agent: stated details reported correctly), shared by the command line and the simulator dashboard.

pull_calls(session_id)   the scout_agent calls of a Langfuse session: the report the scout agent returned, and the coordinator's saved
                         conversation (the person's messages, the questions asked, the person's answers) at the moment of the call
evaluate(calls)          an LLM reads that conversation only and writes the three details the person gave; compare.py then compares them
                         with the scout agent's report (match.py decides whether two values are the same thing)
first_call(calls)        Eval 1 checks only the scout agent's FIRST call of a session (the one for the person's first query)
run_session(session_id)  pull, keep the first call, evaluate, and save in the files below

Files in this folder: scout_calls.json (the calls), eval_results.json (one result per call), session_runs.json (when each session was evaluated),
labels.json (the person's labels, never written here). Nothing is cached: every run asks the models again.
"""
import base64, json, re, threading, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

from compare import compare
from match import Matcher

HERE = Path(__file__).parent
LOCK = threading.Lock()
ENV = {}
for _line in (HERE.parents[1] / 'Place_Discover_Negotiating-AGENT' / '.env').read_text(encoding='utf-8').splitlines():
    if '=' in _line and not _line.startswith('#'):
        _k, _v = _line.split('=', 1); ENV[_k.strip()] = _v.strip().strip('"\'')

if ENV.get('GEMINI_API_KEY'):
    URL, KEY, MODEL = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', ENV['GEMINI_API_KEY'], ENV.get('JUDGE_MODEL', 'gemini-2.5-pro')
else:  # a different family from the scout agent's gpt-oss-120b
    URL, KEY, MODEL = 'https://api.groq.com/openai/v1/chat/completions', ENV['GROQ_API_KEY_2'], ENV.get('JUDGE_MODEL', 'qwen/qwen3.8-27b')
LANGFUSE = (ENV.get('LANGFUSE_BASE_URL') or ENV.get('LANGFUSE_HOST')).rstrip('/')
AUTH = 'Basic ' + base64.b64encode(f"{ENV['LANGFUSE_PUBLIC_KEY']}:{ENV['LANGFUSE_SECRET_KEY']}".encode()).decode()
PROMPT_LABEL = 'draft'
_prompt = None
_match_prompt = None
_matcher = None


def _read(name, default):
    try:
        return json.loads((HERE / name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _write(name, data):
    tmp = HERE / (name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(HERE / name)


def _observations(query):
    for attempt in range(6):
        try:
            req = urllib.request.Request(f'{LANGFUSE}/api/public/v2/observations?{urllib.parse.urlencode(query)}', headers={'Authorization': AUTH})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503):
                time.sleep(4 * (attempt + 1)); continue
            raise
    raise RuntimeError('Langfuse kept refusing')


def _parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _paged(flt):
    rows, cursor = [], None
    while True:
        q = {'filter': json.dumps(flt), 'fromStartTime': '2026-10-01T00:00:00Z', 'toStartTime': '2027-12-31T00:00:00Z', 'limit': 100, 'fields': 'core,basic,io'}
        if cursor:
            q['cursor'] = cursor
        page = _observations(q)
        rows += page['data']
        cursor = (page.get('meta') or {}).get('cursor')
        if not cursor:
            return rows


def _text(content):
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def person_memory(messages):
    """What the coordinator knew from the person: their messages, the questions put to them and their answers. Nothing the coordinator or the
    scout agent produced (its scout_agent requests, the scout reports, its own replies), so the reading cannot lean on them."""
    lines = []
    for m in messages:
        role, content = m.get('role'), m.get('content')
        if role == 'user':
            lines.append(f'Person: {_text(content)}')
        elif role == 'assistant':
            for call in m.get('tool_calls') or []:
                if call.get('name') == 'ask_user':
                    lines.append(f'Coordinator asked the person: "{(call.get("args") or {}).get("question")}"')
        elif role == 'tool' and m.get('name') == 'ask_user' and not isinstance(content, dict):  # a dict is the tool's definition, not an answer
            lines.append(f'Person answered: {_text(content)}')
    return '\n'.join(lines)


def _memories(session_id=None):
    """session id -> [[request, person_memory, used]], oldest first: the coordinator's conversation at each scout_agent call it made."""
    # any model call that decided to call scout_agent: only the coordinator has that tool (older traces name these calls "ChatGroq")
    flt = [{'type': 'string', 'column': 'type', 'operator': '=', 'value': 'GENERATION'}]
    if session_id:
        flt.append({'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': session_id})
    found = {}
    for o in sorted(_paged(flt), key=lambda o: o['startTime']):
        out, inp = _parse(o.get('output')), _parse(o.get('input'))
        if not isinstance(out, dict) or not isinstance(inp, list):
            continue
        for call in out.get('tool_calls') or []:
            if call.get('name') == 'scout_agent':
                req = (call.get('args') or {}).get('request')
                found.setdefault(o.get('sessionId'), []).append([req if isinstance(req, str) else json.dumps(req, ensure_ascii=False), person_memory(inp), False])
    return found


def pull_calls(session_id=None):
    """The scout_agent calls in Langfuse, oldest first: one session, or all of them."""
    flt = [{'type': 'string', 'column': 'name', 'operator': '=', 'value': 'scout_agent'}, {'type': 'string', 'column': 'type', 'operator': '=', 'value': 'TOOL'}]
    if session_id:
        flt.append({'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': session_id})
    memories = _memories(session_id)
    calls = []
    for o in sorted(_paged(flt), key=lambda o: o['startTime']):
        out, inp = _parse(o.get('output')), _parse(o.get('input'))
        content = _parse(out.get('content')) if isinstance(out, dict) and 'content' in out else out  # the report itself, or wrapped as the content of a tool message
        request = inp.get('request') if isinstance(inp, dict) else o.get('input')
        as_text = request if isinstance(request, str) else json.dumps(request, ensure_ascii=False)
        memory = None
        for entry in memories.get(o.get('sessionId'), []):  # the model call that made this tool call: same request, the first one not used yet
            if not entry[2] and entry[0] == as_text:
                entry[2], memory = True, entry[1]
                break
        calls.append({'observation_id': o['id'], 'session_id': o.get('sessionId'), 'time': o['startTime'], 'level': o.get('level'), 'status_message': o.get('statusMessage'),
                      'request': request, 'memory': memory,
                      'report': content if isinstance(content, dict) else None, 'raw_output': None if isinstance(content, dict) else content})
    return calls


def _ask(prompt):
    body = json.dumps({'model': MODEL, 'temperature': 0, 'messages': [{'role': 'user', 'content': prompt}]}).encode()
    for attempt in range(10):
        try:
            req = urllib.request.Request(URL, body, {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'User-Agent': 'scout-eval/1.0'})
            text = json.load(urllib.request.urlopen(req, timeout=120))['choices'][0]['message']['content']
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
            return json.loads(text[text.index('{'): text.rindex('}') + 1])
        except urllib.error.HTTPError as e:
            if e.code == 429 and int(float(e.headers.get('retry-after') or 0)) > 120:  # a daily limit: waiting here would only look like a hang
                raise RuntimeError(f'judge model limit reached, try again in {int(float(e.headers["retry-after"])) // 60 + 1} min')
            if e.code in (429, 500, 502, 503):
                time.sleep(min(60, 8 * (attempt + 1))); continue
            raise
        except ValueError:
            time.sleep(1)  # not JSON: ask again
    return None


def _langfuse_prompt(name):
    from langfuse import Langfuse
    client = Langfuse(public_key=ENV['LANGFUSE_PUBLIC_KEY'], secret_key=ENV['LANGFUSE_SECRET_KEY'], base_url=LANGFUSE)
    return client.get_prompt(name, type='text', label=PROMPT_LABEL, cache_ttl_seconds=0)


def extract_prompt():
    """The extractor prompt, from Langfuse prompt management."""
    global _prompt
    if _prompt is None:
        _prompt = _langfuse_prompt('scout-fields-extract')
    return _prompt


def match_prompt():
    """The prompt of the judge-model step of match.py, from Langfuse prompt management."""
    global _match_prompt
    if _match_prompt is None:
        _match_prompt = _langfuse_prompt('scout-fields-match')
    return _match_prompt


def matcher():
    global _matcher
    if _matcher is None:
        _matcher = Matcher(_ask, match_prompt)
    return _matcher


def evaluate(calls, on_each=None):
    """One result per call. Nothing is cached: the extractor and the matcher are asked again on every run."""
    prompt, out = extract_prompt(), []
    for c in calls:
        request = c['request'] if isinstance(c['request'], str) else json.dumps(c['request'], ensure_ascii=False)
        memory = c.get('memory')
        read = _ask(prompt.compile(memory=memory)) if memory and c['report'] is not None else None
        ref = {k: read.get(k) for k in ('business_type', 'area', 'city', 'missing')} if read else None
        verdict = compare(ref, c['report'], matcher())
        if memory is None:
            verdict = {'verdict': 'unknown', 'failures': ['no_memory'], 'fields': {}, 'matched_by': {},
                       'notes': ["the coordinator's saved conversation for this call was not found in Langfuse"]}
        out.append({'observation_id': c['observation_id'], 'session_id': c['session_id'], 'time': c['time'], 'request': request, 'memory': memory, 'report': c['report'],
                    'reference': ref, 'reference_reasoning': read.get('reasoning') if read else None, **verdict})
        if on_each:
            on_each(out[-1])
    return out


def first_call(calls):
    """Only the scout agent's first call of the session is evaluated: the one made for the person's first query (calls are from one session)."""
    return sorted(calls, key=lambda c: c['time'])[:1]


def save(calls, results, session_id=None):
    """Merge into the files by observation id; remember that this session was evaluated. A session that is evaluated again replaces its earlier results."""
    with LOCK:
        for name, new in (('scout_calls.json', calls), ('eval_results.json', results)):
            merged = {x['observation_id']: x for x in _read(name, []) if not (session_id and x.get('session_id') == session_id)}
            merged.update({x['observation_id']: x for x in new})
            _write(name, sorted(merged.values(), key=lambda x: x['time']))
        if session_id:
            runs = _read('session_runs.json', {})
            runs[session_id] = {'ran_at': time.time(), 'calls': len(calls), 'model': MODEL}
            _write('session_runs.json', runs)


def run_session(session_id):
    """Evaluate the first scout call of one session. A session with no scout call is recorded as such."""
    calls = first_call(pull_calls(session_id))
    results = evaluate(calls)
    save(calls, results, session_id)
    return results
