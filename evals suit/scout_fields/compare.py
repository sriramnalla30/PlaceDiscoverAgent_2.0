"""Eval 1, last step: compare what the scout agent reported with what the person said. Plain code; the matching of two values is match.py.

reference = business_type / area / city, read by an LLM from the coordinator's saved conversation up to the scout agent's FIRST call of the session (see prompts/scout-fields-extract.md)
report    = the scout agent's report for that call (only its business_type, area and city are looked at: this eval stops before the map lookup)

PASS    all three details match.   FAIL   at least one detail does not match.
UNKNOWN nothing could be compared (no report, scout failed, the extractor or the matcher did not answer) and no detail failed.
A detail matches when both are empty, or both have a value and match.py says they are the same thing.
Why a detail fails: dropped (said by the person, null in the report), invented (not said, filled in the report), wrong (a different value).
"""
FIELDS = ('business_type', 'area', 'city')
EMPTY = (None, '')


def _empty(value):
    return value in EMPTY or not str(value).strip()


def compare(reference, report, same):
    """same(field, a, b) -> (True | False | None, how). Returns verdict, failures, notes, per-field status and how each match was decided."""
    if not isinstance(report, dict):
        return {'verdict': 'unknown', 'failures': ['no_report'], 'notes': ['the scout agent returned no report'], 'fields': {}, 'matched_by': {}}
    if not isinstance(reference, dict):
        return {'verdict': 'unknown', 'failures': ['extractor_failed'], 'notes': ['the extractor model did not answer (usually its rate limit); run this session again'], 'fields': {}, 'matched_by': {}}
    if report.get('status') == 'failed':
        return {'verdict': 'unknown', 'failures': ['scout_failed'], 'notes': [f"scout returned status failed: {report.get('problem')}"], 'fields': {}, 'matched_by': {}}
    failures, notes, fields, how_by = [], [], {}, {}
    for f in FIELDS:
        ref, got = reference.get(f), report.get(f)
        if _empty(ref) and _empty(got):
            fields[f] = 'ok'
        elif not _empty(ref) and _empty(got):
            fields[f] = 'dropped'; failures.append('dropped_field'); notes.append(f'{f}: the person said "{ref}" but the report has null')
        elif _empty(ref):
            fields[f] = 'invented'; failures.append('invented_field'); notes.append(f'{f}: the person did not say it but the report has "{got}"')
        else:
            result, how = same(f, ref, got)
            how_by[f] = how
            if result is None:
                fields[f] = 'unknown'; notes.append(f'{f}: "{ref}" vs "{got}" could not be decided ({how})')
            elif result:
                fields[f] = 'ok'
                if str(ref).strip() != str(got).strip():
                    notes.append(f'{f}: "{got}" is the same as "{ref}" ({how})')
            else:
                fields[f] = 'wrong'; failures.append('wrong_value'); notes.append(f'{f}: the person said "{ref}" but the report has "{got}" ({how})')
    failures = sorted(set(failures), key=['dropped_field', 'invented_field', 'wrong_value'].index)
    verdict = 'fail' if failures else 'unknown' if 'unknown' in fields.values() else 'pass'
    return {'verdict': verdict, 'failures': failures, 'notes': notes, 'fields': fields, 'matched_by': how_by}
