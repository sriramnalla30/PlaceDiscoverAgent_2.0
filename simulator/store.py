"""Chat history of the simulated user, one JSON file per chat in ./data/chats/ (copied from the app's store)."""
import json
import os
import tempfile
from pathlib import Path

DATA_DIR = Path(os.getenv('SIM_DATA_DIR') or Path(__file__).resolve().parent / 'data')
MAX_BYTES = 2_000_000
MAX_RUNS = 500


def _dir() -> Path:
    path = DATA_DIR / 'chats'
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_chats() -> list[dict]:
    out = []
    for file in _dir().glob('*.json'):
        try:
            chat = json.loads(file.read_text('utf-8'))
            out.append({'id': file.stem, 'title': str(chat.get('title', 'Chat')), 'updated': float(chat.get('updated', 0))})
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda c: c['updated'], reverse=True)


def load_chat(chat_id: str) -> dict | None:
    try:
        chat = json.loads((_dir() / f'{chat_id}.json').read_text('utf-8'))
    except (OSError, ValueError):
        return None
    return {'id': chat_id, 'title': str(chat.get('title', 'Chat')), 'updated': float(chat.get('updated', 0)), 'log': chat.get('log', [])}


def save_chat(chat_id: str, title: str, updated: float, log: list) -> None:
    data = json.dumps({'title': title[:80], 'updated': updated, 'log': log}, ensure_ascii=False)
    if len(data.encode('utf-8')) > MAX_BYTES:
        raise ValueError('chat is too large to save')
    fd, tmp = tempfile.mkstemp(dir=_dir(), suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(data)
    os.replace(tmp, _dir() / f'{chat_id}.json')


def delete_chat(chat_id: str) -> None:
    (_dir() / f'{chat_id}.json').unlink(missing_ok=True)


def _read(name: str, default):
    try:
        return json.loads((DATA_DIR / name).read_text('utf-8'))
    except (OSError, ValueError):
        return default


def _write(name: str, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, DATA_DIR / name)


def load_results() -> dict:
    """The latest run of each case, for the dashboard cards."""
    return _read('results.json', {})


def load_runs() -> list:
    """Every run, oldest first."""
    runs = _read('runs.json', [])
    return runs if isinstance(runs, list) else []


def save_run(record: dict) -> None:
    results = load_results()
    results[record['case_id']] = record
    _write('results.json', results)
    _write('runs.json', (load_runs() + [record])[-MAX_RUNS:])
