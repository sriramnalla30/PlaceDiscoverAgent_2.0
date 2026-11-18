"""Does match.py tell "the same thing" from "a different thing"? Runs the real steps (tidy text, embedding model, judge model) on labelled pairs and
prints which step decided each one. The pairs are kept apart from the examples written in prompts/scout-fields-match.md.
  python test_match.py            (the judge model is called for the pairs the first two steps cannot decide; nothing is cached)"""
import sys
import eval_core
from match import similarity

sys.stdout.reconfigure(encoding='utf-8')
# (field, A = what the person said, B = what the scout agent reported, expected: same?)
PAIRS = [
    ('city', 'Bangalore', 'bangalore', 1), ('city', 'Bengaluru', 'Bangalore', 1), ('city', 'Bombay', 'Mumbai', 1), ('city', 'Madras', 'Chennai', 1),
    ('city', 'Calcutta', 'Kolkata', 1), ('city', 'Poona', 'Pune', 1), ('city', 'Trivandrum', 'Thiruvananthapuram', 1), ('city', 'बैंगलोर', 'Bangalore', 1),
    ('city', 'हैदराबाद', 'Hyderabad', 1), ('city', 'Hyderabad', 'Hyderabad, Telangana', 1), ('city', 'Delhi', 'New Delhi', 1), ('city', 'Mysore', 'Mysuru', 1),
    ('city', 'Cochin', 'Kochi', 1), ('city', 'Baroda', 'Vadodara', 1), ('city', 'Vijayawada', 'Bezawada', 1),
    ('city', 'Delhi', 'Dehradun', 0), ('city', 'Chennai', 'Mumbai', 0), ('city', 'Pune', 'Punjab', 0), ('city', 'Salem', 'Selam Nagar', 0),
    ('city', 'Vijayawada', 'Visakhapatnam', 0), ('city', 'Mumbai', 'Navi Mumbai', 0), ('city', 'Hyderabad', 'Secunderabad', 0), ('city', 'Bangalore', 'Mangalore', 0),
    ('area', 'Indiranagar', 'indiranagar', 1), ('area', 'Indira Nagar', 'Indiranagar', 1), ('area', 'AYODHYANAGAR', 'Ayodhyanagar', 1), ('area', 'Kukatpally', 'Kukatpalli', 1),
    ('area', 'Koramangala', 'Koramangla', 1), ('area', 'HSR Layout', 'H.S.R. Layout', 1), ('area', 'इंदिरानगर', 'Indiranagar', 1), ('area', 'Banjara Hills', 'Banjarahills', 1),
    ('area', 'Jayanagar', 'Jaynagar', 1), ('area', 'Whitefield', 'White Field', 1), ('area', 'Benz Circle', 'Benz circle', 1),
    ('area', 'Indiranagar', 'Jayanagar', 0), ('area', 'Anna Nagar', 'Adyar', 0), ('area', 'Banjara Hills', 'Jubilee Hills', 0), ('area', 'Whitefield', 'Marathahalli', 0),
    ('area', 'Ayodhyanagar', 'Ashoknagar', 0), ('area', 'Vijayanagar', 'Jayanagar', 0), ('area', 'Rajajinagar', 'Rajendra Nagar', 0), ('area', 'Koramangala', 'Kormangala Extension', 0),
    ('area', 'Indiranagar', 'Indindra', 0), ('area', 'Madhapur', 'Mahadevapura', 0),
    ('business_type', 'gyms', 'gym', 1), ('business_type', 'pharmacies', 'pharmacy', 1), ('business_type', 'restaurants', 'restarants', 1), ('business_type', 'cafes', 'cafe', 1),
    ('business_type', 'Dentists', 'dentist', 1), ('business_type', 'pharmacies', 'medical shops', 1), ('business_type', 'gyms', 'fitness centers', 1), ('business_type', 'bakeries', 'bakery', 1),
    ('business_type', 'salons', 'beauty parlours', 1), ('business_type', 'hotels', 'hotel', 1),
    ('business_type', 'gyms', 'pharmacies', 0), ('business_type', 'cafes', 'restaurants', 0), ('business_type', 'dentists', 'doctors', 0), ('business_type', 'hotels', 'hostels', 0),
    ('business_type', 'bakeries', 'cafes', 0), ('business_type', 'salons', 'spas', 0), ('business_type', 'pharmacies', 'hospitals', 0), ('business_type', 'restaurants', 'bars', 0),
]

matcher = eval_core.matcher()
wrong, by = [], {}
for field, a, b, want in PAIRS:
    got, how = matcher(field, a, b)
    step = 'embedding model' if how.startswith('embedding') else 'judge model' if how.startswith('judge') else how
    by[step] = by.get(step, 0) + 1
    ok = got is not None and bool(got) == bool(want)
    if not ok:
        wrong.append((field, a, b, want, got, how))
    cos = similarity(a, b)
    print(f'{"ok   " if ok else "WRONG"} {field:13} {a!r:22} ~ {b!r:24} want {"same" if want else "diff"}  got {got!s:5}  cos={cos if cos is None else round(cos, 2)}  by {how}')
print(f'\n{len(PAIRS) - len(wrong)}/{len(PAIRS)} right; decided by: {by}')
for w in wrong:
    print('  wrong:', w)
