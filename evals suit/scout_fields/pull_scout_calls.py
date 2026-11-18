"""Pull every scout_agent call from Langfuse into scout_calls.json (read-only on Langfuse). Uses the v2 observations API."""
import eval_core

calls = eval_core.pull_calls()
eval_core.save(calls, [])
print(len(calls), 'scout calls;', len({c['session_id'] for c in calls}), 'sessions;', sum(1 for c in calls if c['report'] is None), 'without a report')
