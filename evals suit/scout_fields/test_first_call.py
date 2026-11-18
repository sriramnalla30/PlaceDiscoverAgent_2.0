"""Eval 1 evaluates only the scout agent's first call of a session.  python test_first_call.py"""
from eval_core import first_call

calls = [{'observation_id': 'c', 'time': '2026-11-17T07:11:50Z'}, {'observation_id': 'a', 'time': '2026-11-17T07:08:15Z'}, {'observation_id': 'b', 'time': '2026-11-17T07:09:00Z'}]
cases = [('several calls: the oldest one', calls, ['a']), ('one call', calls[:1], ['c']), ('no call', [], [])]
bad = 0
for name, given, want in cases:
    got = [c['observation_id'] for c in first_call(given)]
    bad += got != want
    print('ok  ' if got == want else 'FAIL', name, '' if got == want else f'-> {got} (wanted {want})')
raise SystemExit(bad > 0)
