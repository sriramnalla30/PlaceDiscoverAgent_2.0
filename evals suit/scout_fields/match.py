"""Eval 1, the matching step: do two values (what the person said / what the scout agent reported) name the same thing?

Regex alone cannot do this ("Bangalore" = "Bengaluru", "Vizag" = "Visakhapatnam", "बैंगलोर" = "Bangalore"), and measuring showed that no
single score can either: with a multilingual embedding model "Gurgaon ~ Gurugram" scores 0.42 but "Indiranagar ~ Jayanagar" scores 0.84.
So three steps, cheapest first. Each step only answers when it is sure:

  1. same text after tidying   case, punctuation, spaces, singular/plural        "HSR Layout" = "h.s.r. layout", "gyms" = "gym"
  2. embedding similarity      a local multilingual model, cosine >= 0.90 means the same (no pair of different places scored that high)
  3. a judge model             everything else: one small question, "same place?", asked fresh every time

An index such as FAISS is not used: it finds the nearest vectors among many, and here exactly two values are compared.
If the embedding model is not installed (fastembed) step 2 is skipped and step 3 decides.
"""
import re, threading

EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # 0.22 GB, 50+ languages, runs on the CPU
EMBED_SAME = 0.90
_embedder, _embed_failed, _embed_lock = None, False, threading.Lock()


def _singular(word):
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes', 'sses')):
        return word[:-2]
    return word[:-1] if word.endswith('s') and not word.endswith('ss') and len(word) > 3 else word


def tidy(value, plural=False):
    """Lower case, no punctuation, no spaces. For business types also singular: "Pizza Places" -> "pizzaplace". None when empty."""
    if value is None or not str(value).strip():
        return None
    words = re.sub(r'[^\w\s]', ' ', str(value).lower(), flags=re.UNICODE).split()
    return (''.join(_singular(w) if plural else w for w in words)) or None


def similarity(a, b):
    """Cosine similarity of the two texts with the local embedding model, or None when the model is not available."""
    global _embedder, _embed_failed
    with _embed_lock:
        if _embed_failed:
            return None
        try:
            if _embedder is None:
                from fastembed import TextEmbedding
                _embedder = TextEmbedding(EMBED_MODEL)
            va, vb = [list(map(float, v)) for v in _embedder.embed([a, b])]
        except Exception:
            _embed_failed = True
            return None
    dot = sum(x * y for x, y in zip(va, vb))
    return dot / ((sum(x * x for x in va) ** .5) * (sum(y * y for y in vb) ** .5))


class Matcher:
    """matcher(field, a, b) -> (True | False | None, how). None means nothing could decide (the judge model did not answer). Nothing is cached."""

    def __init__(self, ask, get_prompt):
        self.ask, self.get_prompt = ask, get_prompt

    def __call__(self, field, a, b):
        plural = field == 'business_type'
        x, y = tidy(a, plural), tidy(b, plural)
        if x == y:
            return True, 'same text'
        score = similarity(str(a), str(b))
        # one text inside the other ("Mumbai" / "Navi Mumbai" vs "Hyderabad" / "Hyderabad, Telangana") scores high either way: the judge model decides
        if score is not None and score >= EMBED_SAME and not (x and y and (x in y or y in x)):
            return True, f'embedding similarity {score:.2f}'
        answer = self.ask(self.get_prompt().compile(field=field, kind='kind of business' if plural else 'place', a=str(a), b=str(b)))
        if not answer or not isinstance(answer.get('same'), bool):
            return None, 'judge model did not answer'
        return answer['same'], 'judge model' + (f' (embedding similarity {score:.2f})' if score is not None else '')
