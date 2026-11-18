"""Save the two judge prompts in Langfuse prompt management. A new version is created only when the text changed.
The files in prompts/ are the source of truth; the runner reads them back from Langfuse by label."""
import sys
from pathlib import Path
from langfuse import Langfuse

HERE = Path(__file__).parent
ENV = {}
for line in (HERE.parents[1] / 'Place_Discover_Negotiating-AGENT' / '.env').read_text(encoding='utf-8').splitlines():
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1); ENV[k.strip()] = v.strip().strip('"\'')
client = Langfuse(public_key=ENV['LANGFUSE_PUBLIC_KEY'], secret_key=ENV['LANGFUSE_SECRET_KEY'], base_url=ENV.get('LANGFUSE_BASE_URL') or ENV['LANGFUSE_HOST'])
label = sys.argv[1] if len(sys.argv) > 1 else 'draft'
for path in sorted((HERE / 'prompts').glob('*.md')):
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
