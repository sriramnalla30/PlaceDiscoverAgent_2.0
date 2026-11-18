"""Eval 1 on the command line: the scout agent's first call of each session only (see eval_core.py).
usage: python run_eval.py <session_id> [<session_id> ...]
       python run_eval.py --all          every session that has a scout call"""
import sys
import eval_core

ids = [a for a in sys.argv[1:] if not a.startswith('--')]
if '--all' in sys.argv:
    ids = sorted({c['session_id'] for c in eval_core.pull_calls() if c['session_id']})
if not ids:
    sys.exit(__doc__)
print(f'model {eval_core.MODEL}; extractor prompt scout-fields-extract v{eval_core.extract_prompt().version}; match prompt scout-fields-match v{eval_core.match_prompt().version}')
for sid in ids:
    results = eval_core.run_session(sid)
    print(f'\nsession {sid}')
    for r in results:
        print(' ', r['verdict'].upper().ljust(8), repr(r['request'])[:60].ljust(62), r['failures'])
        for n in r['notes']:
            print('      ', n)
