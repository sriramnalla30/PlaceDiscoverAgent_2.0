"""Save prompts/handoff-judge.md in Langfuse prompt management. A new version is created only when the text changed.
The file is the source of truth; handoff_eval.py reads the prompt back from Langfuse by label.
usage: python sync_prompt.py [label]     (default label: draft)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scout_fields'))
import eval_core
from langfuse import Langfuse

label = sys.argv[1] if len(sys.argv) > 1 else 'draft'
client = Langfuse(public_key=eval_core.ENV['LANGFUSE_PUBLIC_KEY'], secret_key=eval_core.ENV['LANGFUSE_SECRET_KEY'], base_url=eval_core.LANGFUSE)
for path in sorted((Path(__file__).parent / 'prompts').glob('*.md')):
    text = path.read_text(encoding='utf-8')
    try:
        current = client.get_prompt(path.stem, type='text', label=label, cache_ttl_seconds=0)
    except Exception:
        current = None
    if current is None or current.prompt != text:
        current = client.create_prompt(name=path.stem, prompt=text, type='text', labels=[label], commit_message=f'synced from prompts/{path.name}')
        print('saved', path.stem, 'version', current.version, 'label', label)
    else:
        print('unchanged', path.stem, 'version', current.version)
client.flush()
