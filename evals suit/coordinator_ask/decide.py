"""Gemini Decisions API (model gemini-decision-1): closed yes/no questions about one input, answered with a confidence and no text.

decide(text, questions)   -> {question name: {'yes': bool, 'confidence': float, 'p_yes': float}}
One request carries all the questions about one input (the input is billed once). The key is GOOGLE_DECISIONS_API_KEY in the app's .env.
"""
import json, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scout_fields'))
import eval_core  # the .env values

URL = 'https://generativelanguage.googleapis.com/v1beta/decisions'
MODEL = 'gemini-decision-1'


def decide(text, questions):
    """questions: {name: instruction}. Each is a yes/no ('noul') question."""
    key = eval_core.ENV.get('GOOGLE_DECISIONS_API_KEY')
    if not key:
        raise RuntimeError('GOOGLE_DECISIONS_API_KEY is not set in the .env file')
    body = json.dumps({'model': MODEL, 'input': text, 'questions': {k: {'type': 'noul', 'instructions': v} for k, v in questions.items()}}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, body, {'x-goog-api-key': key, 'Content-Type': 'application/json'})
            answers = json.load(urllib.request.urlopen(req, timeout=60))['answers']
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(min(30, 4 * (attempt + 1))); continue
            raise RuntimeError(f'Decisions API error {e.code}: {e.read()[:200].decode("utf-8", "replace")}')
    out = {}
    for name in questions:
        p_yes = float(answers[name]['noul'])  # 'noul' is the probability of yes; 'confidence' is the probability of the answer it leans to
        out[name] = {'yes': p_yes >= 0.5, 'confidence': max(p_yes, 1 - p_yes), 'p_yes': p_yes}
    return out
