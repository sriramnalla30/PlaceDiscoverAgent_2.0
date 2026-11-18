"""The cases labelled by hand, run through the judge: python calibrate.py
Each case is (name, conversation, channel, message, whether the search outcome is in the conversation, the label). Calls the Decisions API.
Cases 1-5 were labelled by the person, 6-10 by me, R1-R5 are the real messages of session e56721ad (real_cases.json) labelled by the person."""
import json
import sys
from pathlib import Path
import ask_eval as A

RESULTS = 'Scout agent reported: status ready; business gyms; area Koramangala; city Bangalore; search found 20 places; best overall Live Fitness Gym (4.9 stars, 1059 reviews)'

CASES = [
    ('1 related extra question', 'Person: restaurants in Indiranagar Bangalore', 'ask', 'Is this for a family dinner or a quick lunch?', None, 'pass'),
    ('2 asks what was already said', 'Person: cafes in Indiranagar Bangalore', 'ask', 'Which area are you interested in?', None, 'fail'),
    ('3 "prepared" only, results exist', f'Person: gyms in Koramangala Bangalore\n{RESULTS}', 'reply', 'Your request is prepared.', 'results', 'fail'),
    ('4 not a place request', "Person: what's the weather in Bangalore today?", 'reply',
     "I can't check the weather, but I can find places like gyms or cafes. What are you looking for?", None, 'pass'),
    ('5 wider area, explained', 'Person: gyms in Koramangala Bangalore\nScout agent reported: status ready; business gyms; area Koramangala; city Bangalore; '
                               'search found 20 places (2 in Koramangala, 18 in nearby HSR Layout)', 'reply',
     'I found only 2 gyms in Koramangala, so I also searched nearby HSR Layout and found 18 more.', 'results', 'pass'),
    # extra cases beyond the five
    ('6 results shown', f'Person: gyms in Koramangala Bangalore\n{RESULTS}', 'reply',
     'I found 20 gyms in Koramangala, Bangalore. The top one is Live Fitness Gym (4.9 stars, 1,059 reviews).', 'results', 'pass'),
    ('7 relays the missing detail', 'Person: find me cafes in Bangalore\nScout agent reported: status needs_input; business cafes; city Bangalore; missing area', 'ask',
     'Which area of Bangalore should I look in?', None, 'pass'),
    ('8 different place type', f'Person: gyms in Koramangala Bangalore\n{RESULTS}', 'reply', 'I found 20 restaurants in Koramangala, Bangalore.', 'results', 'fail'),
    ('9 different city', 'Person: pharmacies in Hyderabad', 'ask', 'Which area of Bangalore are you in?', None, 'fail'),
    ('10 says what went wrong', 'Person: gyms in Xyzville Bangalore\nScout agent reported: status failed; problem: no map location found for "Xyzville"', 'reply',
     'I could not find a place called Xyzville in Bangalore. Could you give a nearby landmark?', 'problem', 'pass'),
]

# the real session: what the person labelled after reading the judge's answers (R2: an honest "I cannot" is helpful; R4: asks what they just said)
for case, label in zip(json.loads((Path(__file__).parent / 'real_cases.json').read_text(encoding='utf-8')), ('pass', 'pass', 'fail', 'fail', 'fail')):
    CASES.append((f'R{case["n"]} real: {case["message"][:26]}', case['memory'], case['kind'], case['message'], case['outcome'], label))

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    wrong = 0
    for name, memory, kind, message, found, label in CASES:
        r = A.judge(memory, kind, message, found)
        got = r['verdict']
        mark = 'ok ' if got == label else 'DIFF'
        wrong += got != label
        print(f'{mark} {name:36} labelled {label:5} judged {got:6} {",".join(r["failures"]):12} {r["scores"]}')
    print(f'\n{len(CASES) - wrong}/{len(CASES)} match the labels')
