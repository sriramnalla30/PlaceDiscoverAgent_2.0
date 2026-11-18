"""Eval 2 without the network: which messages are read, what the judge is shown, how the answers become a verdict.  python test_events.py"""
import ask_eval as A

# only a coordinator's question and its final reply are read: reasoning, the note beside a tool call and the scout agent's own calls are not
tools = [{'role': 'tool', 'content': {'type': 'function', 'function': {'name': 'scout_agent'}}}]
report = {'status': 'ready', 'business_type': 'gyms', 'area': 'Koramangala', 'city': 'Bangalore', 'search': {'count': 20, 'highlights': {'best': {'name': 'Live Fitness Gym', 'rating': 4.9, 'reviews': 1059}}}}
memory = tools + [{'role': 'user', 'content': 'gyms in Koramangala Bangalore'},
                  {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'scout_agent', 'args': {'request': 'gyms in Koramangala Bangalore'}}]},
                  {'role': 'tool', 'name': 'scout_agent', 'content': report}]
gen = lambda t, name, inp, out: {'startTime': t, 'name': name, 'input': inp, 'output': out}
gens = [gen('1', 'ChatGroq', tools, {'role': 'assistant', 'content': 'thinking aloud', 'tool_calls': [{'name': 'scout_agent', 'args': {}}]}),
        gen('2', 'Scout agent: model call', [{'role': 'tool', 'content': {'type': 'function', 'function': {'name': 'Report'}}}], {'role': 'assistant', 'content': 'scout talk'}),
        gen('3', 'ChatGroq', memory, {'role': 'assistant', 'content': 'Found 20 gyms.', 'additional_kwargs': {'reasoning_content': 'secret'}})]
events = A.build_events(gens)
assert [(e['kind'], e['text']) for e in events] == [('reply', 'Found 20 gyms.')], events
assert events[0]['outcome'] == 'results'
assert 'Person: gyms in Koramangala Bangalore' in events[0]['memory'] and 'search found 20 places; best overall Live Fitness Gym' in events[0]['memory']
assert 'secret' not in events[0]['memory'] and 'thinking aloud' not in events[0]['memory']

# an ask_user question is a message too
ask = [gen('1', 'ChatGroq', tools, {'role': 'assistant', 'content': 'note', 'tool_calls': [{'name': 'ask_user', 'args': {'question': 'Which area?'}}]})]
assert [(e['kind'], e['text']) for e in A.build_events(ask)] == [('ask', 'Which area?')]

# what the scout agent reported decides whether "show the results" applies
rep = lambda r: [{'role': 'tool', 'name': 'scout_agent', 'content': r}]
assert A.outcome(rep(report)) == 'results'
assert A.outcome(rep({'status': 'failed', 'problem': 'x'})) == 'problem'
assert A.outcome(rep({'status': 'ready', 'args': {}})) is None            # an older run: arguments only, no search
assert A.outcome(rep({'status': 'needs_input', 'missing': ['area']})) is None
assert A.outcome([]) is None

# the verdict: a sure yes fails, an unsure answer sends it to review, a sure no passes; the extra question is asked only after a search outcome
asked = []
def fake(answers):
    def decide(text, questions):
        asked.append(set(questions))
        return {q: {'yes': p >= .5, 'confidence': max(p, 1 - p), 'p_yes': p} for q, p in ((q, answers.get(q, 0.02)) for q in questions)}
    return decide
for answers, found, want, fails in (({}, None, 'pass', []), ({'asks_known': .93}, None, 'fail', ['asks_known']), ({'off_topic': .6}, None, 'review', []),
                                    ({'no_results': .95}, 'results', 'fail', ['no_results']), ({'no_results': .95}, None, 'pass', [])):
    A.decide = fake(answers)
    got = A.judge('Person: x', 'reply', 'm', found)
    assert (got['verdict'], got['failures']) == (want, fails), (answers, got)
assert asked[0] == {'off_topic', 'asks_known', 'unhelpful'} and 'no_results' in asked[3] and 'no_results' not in asked[4]
A.decide = fake({'no_results': .95})
assert A.judge('Person: x', 'ask', 'm', 'results')['verdict'] == 'pass'    # a question is never asked to show results
assert A.session_verdict([]) == 'no messages'
print('ok')
