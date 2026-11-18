# Starts the simulator. Uses the app's virtual environment when it exists (it already has fastapi, uvicorn and httpx).
$app = Join-Path $PSScriptRoot '..\Place_Discover_Negotiating-AGENT\.venv\Scripts\python.exe'
$python = if (Test-Path $app) { $app } else { 'python' }
& $python (Join-Path $PSScriptRoot 'server.py')
