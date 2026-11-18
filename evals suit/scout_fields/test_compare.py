"""Unit tests for compare.py (no network). The matching of two values is replaced by a small fake, so only the PASS / FAIL logic is tested.
  python test_compare.py"""
from compare import compare

SAME = {('city', 'Bengaluru', 'Bangalore'), ('city', 'Vizag', 'Visakhapatnam'), ('area', 'इंदिरानगर', 'Indiranagar')}
CALLS = []


def fake_same(field, a, b):
    CALLS.append((field, a, b))
    if a.strip().lower() == b.strip().lower():
        return True, 'same text'
    if (field, a, b) in SAME:
        return True, 'judge model'
    if (field, a, b) == ('area', 'Koramangala', '??'):
        return None, 'judge model did not answer'
    return False, 'judge model'


def R(status='ready', b=None, a=None, c=None):
    return dict(status=status, business_type=b, area=a, city=c)


def ref(b=None, a=None, c=None):
    return dict(business_type=b, area=a, city=c)


CASES = [
    ('all three the same', ref('gyms', 'Koramangala', 'Bangalore'), R('ready', 'gyms', 'Koramangala', 'Bangalore'), 'pass', []),
    ('original bug: everything null', ref('restaurants', None, 'Bangalore'), R('needs_input'), 'fail', ['dropped_field']),
    ('partial request done right', ref('restaurants', None, 'Bangalore'), R('needs_input', 'restaurants', None, 'Bangalore'), 'pass', []),
    ('one word', ref('gyms'), R('needs_input', 'gyms'), 'pass', []),
    ('person gave city and area, scout kept only the business', ref('pharmacies', 'AYODHYANAGAR', 'vijayawada'), R('needs_input', 'pharmacies'), 'fail', ['dropped_field']),
    ('case only differs', ref('dentists', 'hsr layout', 'bangalore'), R('ready', 'dentists', 'HSR Layout', 'Bangalore'), 'pass', []),
    ('Bengaluru is Bangalore (matcher says same)', ref('cafes', 'Indiranagar', 'Bengaluru'), R('ready', 'cafes', 'Indiranagar', 'Bangalore'), 'pass', []),
    ('other script (matcher says same)', ref('cafes', 'इंदिरानगर', 'Bangalore'), R('ready', 'cafes', 'Indiranagar', 'Bangalore'), 'pass', []),
    ('wrong city', ref('gyms', 'Koramangala', 'Bangalore'), R('ready', 'gyms', 'Koramangala', 'Mysore'), 'fail', ['wrong_value']),
    ('cut-off area', ref('cafes', 'Indiranagar', 'Bangalore'), R('needs_input', 'cafes', 'Indindra', 'Bangalore'), 'fail', ['wrong_value']),
    ('invented city', ref('pharmacies', 'Whitefield'), R('ready', 'pharmacies', 'Whitefield', 'Bangalore'), 'fail', ['invented_field']),
    ('near me as area', ref('gyms', None, 'Chennai'), R('ready', 'gyms', 'near me', 'Chennai'), 'fail', ['invented_field']),
    ('two details fail', ref('cafes', 'Indiranagar', 'Bangalore'), R('needs_input', None, 'Indindra', 'Bangalore'), 'fail', ['dropped_field', 'wrong_value']),
    ('not a place request', ref(), R('needs_input'), 'pass', []),
    ('missing list and status are not judged here', ref('gyms', 'Koramangala'), dict(status='ready', business_type='gyms', area='Koramangala', city=None, missing=['area']), 'pass', []),
    ('map lookup failure is not a field error', ref('cafes', 'Qzxvbnmlk', 'Vijayawada'), dict(status='needs_input', business_type='cafes', area='Qzxvbnmlk', city='Vijayawada', missing=['area'], question='Another spelling?'), 'pass', []),
    ('matcher cannot decide', ref('gyms', 'Koramangala', 'Bangalore'), R('ready', 'gyms', '??', 'Bangalore'), 'unknown', []),
    ('a fail beats an undecided detail', ref('gyms', 'Koramangala', 'Bangalore'), R('ready', None, '??', 'Bangalore'), 'fail', ['dropped_field']),
    ('no report', ref('gyms'), None, 'unknown', ['no_report']),
    ('scout failed', ref('gyms'), dict(status='failed', problem='x'), 'unknown', ['scout_failed']),
    ('extractor gave no reading', None, R('ready', 'gyms', 'Koramangala', 'Bangalore'), 'unknown', ['extractor_failed']),
]

bad = 0
for name, r, report, verdict, failures in CASES:
    got = compare(r, report, fake_same)
    ok = got['verdict'] == verdict and got['failures'] == failures
    bad += not ok
    print('ok  ' if ok else 'FAIL', name, '' if ok else f"-> {got['verdict']} {got['failures']} (wanted {verdict} {failures}) {got['notes']}")
print(f'{len(CASES) - bad}/{len(CASES)} passed')
raise SystemExit(bad > 0)
