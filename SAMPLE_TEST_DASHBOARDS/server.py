"""Local Maps Scraper laboratory. Python standard library only.

Runs the gosom google-maps-scraper binary in fast mode, one query per click, and shows
what it returns. No API key and no LLM: the scraper reads Google Maps' own search request.
"""
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get('GMAPS_BIN', ROOT.parent / 'test_check' / 'bin' / 'gmaps.exe'))
HISTORY_DB = Path(os.environ.get('MAPS_HISTORY_DB', ROOT / 'history' / 'maps_runs.sqlite3'))
RUN_TIMEOUT = 90
RUN_LOCK = threading.Lock()  # one scrape at a time: be gentle with Google

AREAS = [
    {'name': 'Koramangala', 'city': 'Bangalore', 'lat': 12.9352, 'lng': 77.6245},
    {'name': 'Indiranagar', 'city': 'Bangalore', 'lat': 12.9784, 'lng': 77.6408},
    {'name': 'HSR Layout', 'city': 'Bangalore', 'lat': 12.9116, 'lng': 77.6389},
    {'name': 'Whitefield', 'city': 'Bangalore', 'lat': 12.9698, 'lng': 77.7500},
    {'name': 'Jayanagar', 'city': 'Bangalore', 'lat': 12.9250, 'lng': 77.5938},
    {'name': 'Vijayawada', 'city': '', 'lat': 16.5062, 'lng': 80.6480},
]
STD_METRO = ('80', '40', '11', '22', '44', '20', '33', '79')


@contextmanager
def history_connection():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB, timeout=15)
    try:
        with connection:
            connection.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, summary TEXT NOT NULL, body TEXT NOT NULL)')
            yield connection
    finally:
        connection.close()


def save_history(result):
    record = {**result, 'id': uuid.uuid4().hex, 'timestamp': datetime.now(timezone.utc).isoformat(), 'history_saved': True}
    summary = {k: record.get(k) for k in ('id', 'timestamp', 'params', 'ok', 'elapsed_ms', 'history_saved', 'metrics')}
    summary['summary_only'] = True
    with history_connection() as connection:
        connection.execute('INSERT INTO runs VALUES (?, ?, ?, ?)', (record['id'], record['timestamp'], json.dumps(summary), json.dumps(record, ensure_ascii=False)))
    return record


def get_history(offset=0):
    with history_connection() as connection:
        total = connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
        rows = connection.execute('SELECT summary FROM runs ORDER BY timestamp DESC, id DESC LIMIT 50 OFFSET ?', (offset,)).fetchall()
    return {'runs': [json.loads(row[0]) for row in rows], 'total': total, 'offset': offset}


def get_run(run_id):
    with history_connection() as connection:
        row = connection.execute('SELECT body FROM runs WHERE id = ?', (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def phone_info(raw):
    """Derived by this dashboard, not by the scraper: kind guess and +91 form for a Google phone string."""
    digits = re.sub(r'\D', '', raw or '')
    if not digits:
        return {'kind': 'none', 'e164': ''}
    if digits.startswith('1800') or digits.startswith('1860'):
        return {'kind': 'toll-free', 'e164': '+91' + digits}
    national = digits[1:] if digits.startswith('0') else digits
    if national.startswith('91') and len(national) == 12:
        national = national[2:]
    if len(national) == 10 and national[0] in '6789' and not national.startswith(STD_METRO):
        return {'kind': 'mobile', 'e164': '+91' + national}
    return {'kind': 'landline', 'e164': '+91' + national}


def enrich(places):
    shared = {}
    for place in places:
        digits = re.sub(r'\D', '', place.get('phone') or '')
        if digits:
            shared[digits] = shared.get(digits, 0) + 1
    for place in places:
        info = phone_info(place.get('phone'))
        digits = re.sub(r'\D', '', place.get('phone') or '')
        place['_phone_kind'] = info['kind']
        place['_phone_e164'] = info['e164']
        place['_phone_shared'] = shared.get(digits, 0) if digits else 0
    return places


def filled(value):
    if value in (None, '', [], 0):
        return False
    if isinstance(value, dict):
        return any(filled(v) for v in value.values())
    return True


def compute_metrics(places):
    count = len(places)
    def n(pred):
        return sum(1 for p in places if pred(p))
    return {
        'places': count,
        'with_phone': n(lambda p: p.get('phone')),
        'with_website': n(lambda p: p.get('web_site')),
        'with_hours': n(lambda p: filled(p.get('open_hours'))),
        'mobile': n(lambda p: p.get('_phone_kind') == 'mobile'),
        'landline': n(lambda p: p.get('_phone_kind') == 'landline'),
        'toll_free': n(lambda p: p.get('_phone_kind') == 'toll-free'),
        'shared_phone_places': n(lambda p: p.get('_phone_shared', 0) > 1),
    }


def validate(payload):
    supplied = payload.get('params', {})
    if not isinstance(supplied, dict):
        raise ValueError('Parameters must be a JSON object.')
    query = str(supplied.get('query', '')).strip()
    if not query:
        raise ValueError('Enter a search query, for example: gyms in Koramangala Bangalore.')
    if len(query) > 200 or '\n' in query or '\r' in query:
        raise ValueError('Use one query line of at most 200 characters.')
    try:
        lat, lng = float(supplied.get('lat')), float(supplied.get('lng'))
    except (TypeError, ValueError):
        raise ValueError('Latitude and longitude are required in fast mode, for example 12.9352 and 77.6245.')
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError('Latitude must be -90..90 and longitude -180..180.')
    params = {'query': query, 'lat': lat, 'lng': lng}
    radius = str(supplied.get('radius', '')).strip()
    if radius:
        if not radius.isdigit() or not 100 <= int(radius) <= 50000:
            raise ValueError('Radius must be a whole number of metres between 100 and 50000.')
        params['radius'] = int(radius)
    lang = str(supplied.get('lang', '')).strip()
    if lang:
        if not re.fullmatch(r'[a-z]{2}(-[A-Za-z]{2,4})?', lang):
            raise ValueError('Language must look like en, hi or kn.')
        params['lang'] = lang
    return params


def build_command(params, query_file, out_file):
    command = [str(BINARY), '-input', str(query_file), '-results', str(out_file), '-json', '-fast-mode', '-c', '1', '-depth', '1',
               '-geo', f"{params['lat']},{params['lng']}", '-exit-on-inactivity', '15s']
    if 'radius' in params:
        command += ['-radius', str(params['radius'])]
    if 'lang' in params:
        command += ['-lang', params['lang']]
    return command


def parse_output(text):
    places = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        # A query whose pin is far from the named city can come back as `null` or a list with null entries.
        places.extend(p for p in (data if isinstance(data, list) else [data]) if isinstance(p, dict))
    return places


def tail_message(log):
    """Last meaningful scraper log lines, without progress bars or banners."""
    lines = [l.strip() for l in log.splitlines() if l.strip() and l[0] not in '|╔║╚']
    errors = [l for l in lines if '"level":"error"' in l or 'required' in l.lower() or 'error' in l.lower()]
    picked = (errors or lines)[-3:]
    out = []
    for l in picked:
        try:
            obj = json.loads(l)
            out.append(obj.get('error') or obj.get('message') or l)
        except json.JSONDecodeError:
            out.append(l)
    return ' | '.join(out)[:600]


def execute(payload):
    params = validate(payload)
    if not BINARY.exists():
        raise ValueError(f'Scraper binary not found at {BINARY}. Set GMAPS_BIN or put gmaps.exe in test_check/bin/.')
    if not RUN_LOCK.acquire(blocking=False):
        raise ValueError('Another scrape is still running. Wait for it to finish.')
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='mapslab_') as work:
            work = Path(work)
            query_file, out_file = work / 'q.txt', work / 'out.json'
            query_file.write_text(params['query'] + '\n', encoding='utf-8')
            command = build_command(params, query_file, out_file)
            try:
                proc = subprocess.run(command, cwd=work, capture_output=True, timeout=RUN_TIMEOUT)
            except subprocess.TimeoutExpired:
                return {'ok': False, 'error_kind': 'timeout', 'message': f'The scraper did not finish within {RUN_TIMEOUT} seconds.', 'params': params, 'elapsed_ms': round((time.monotonic() - started) * 1000)}
            log = (proc.stdout + proc.stderr).decode('utf-8', errors='replace')
            raw = out_file.read_text(encoding='utf-8-sig', errors='replace') if out_file.exists() else ''
    finally:
        RUN_LOCK.release()
    elapsed = round((time.monotonic() - started) * 1000)
    shown_command = ['gmaps', '-fast-mode', '-json', '-c', '1', '-geo', f"{params['lat']},{params['lng']}"] + (['-radius', str(params['radius'])] if 'radius' in params else []) + (['-lang', params['lang']] if 'lang' in params else []) + ['-input', 'q.txt']
    try:
        places = enrich(parse_output(raw))
    except json.JSONDecodeError:
        return {'ok': False, 'error_kind': 'parse', 'message': 'The scraper wrote output that is not valid JSON.', 'log': tail_message(log), 'params': params, 'elapsed_ms': elapsed, 'command': shown_command}
    ok = proc.returncode == 0 and bool(places)
    result = {'ok': ok, 'params': params, 'elapsed_ms': elapsed, 'command': shown_command, 'places': places, 'metrics': compute_metrics(places), 'raw': raw, 'log': tail_message(log)}
    if not places:
        result['error_kind'] = 'empty'
        result['message'] = ('No places came back. Fast mode can only read list searches such as "gyms in Koramangala Bangalore". '
                             'A query that names one specific business is treated by Google as a single place and returns nothing. '
                             'Also check that the latitude/longitude pin is in the same city as the query.')
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, data, status=200, mime='application/json; charset=utf-8'):
        body = json.dumps(data, ensure_ascii=False).encode() if mime.startswith('application/json') else data
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/config':
            return self.send({'binary_found': BINARY.exists(), 'areas': AREAS})
        if path == '/api/history':
            try:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                return self.send(get_history(max(0, int(query.get('offset', ['0'])[0]))))
            except ValueError:
                return self.send({'error': 'Invalid history offset.'}, 400)
        if path.startswith('/api/history/'):
            record = get_run(path.rsplit('/', 1)[-1])
            return self.send(record if record else {'error': 'Saved run not found.'}, 200 if record else 404)
        files = {'/': ('index.html', 'text/html'), '/maps_graph': ('maps_graph.html', 'text/html'),
                 '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if path not in files:
            return self.send({'error': 'Not found'}, 404)
        name, mime = files[path]
        return self.send((ROOT / 'static' / name).read_bytes(), mime=mime + '; charset=utf-8')

    def do_POST(self):
        # Disallow cross-origin pages from triggering scrapes on this machine.
        if self.headers.get('Origin') and self.headers['Origin'] != f'http://{self.headers.get("Host")}':
            return self.send({'error': 'Cross-origin request rejected.'}, 403)
        if self.path != '/api/search':
            return self.send({'error': 'Not found'}, 404)
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send({'error': 'Send application/json.'}, 415)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 16384:
                raise ValueError('Request must be between 1 byte and 16 KB.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Request body must be an object.')
            result = execute(body)
            try:
                result = save_history(result)
            except (OSError, sqlite3.Error):
                result['history_saved'] = False
                result['history_warning'] = 'Run finished, but saving to disk failed. Export this run to keep a copy.'
            self.send(result)
        except (ValueError, TypeError) as error:
            self.send({'error': str(error)}, 400)
        except Exception:
            self.send({'error': 'Local server could not process this run.'}, 500)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Maps Scraper Lab: http://127.0.0.1:{args.port}', flush=True)
    print(f'Scraper binary: {BINARY} ({"found" if BINARY.exists() else "NOT FOUND"})', flush=True)
    print('No API key needed. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
