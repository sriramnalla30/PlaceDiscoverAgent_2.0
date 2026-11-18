# Scout simulator

A fake user that drives the real Scout chat page. Three pages:

1. **Test cases** (`/`): 5 happy-flow and 5 ambiguity cases, each with the messages the fake user will type, what is expected, and its last result. Press **Run** on a card for one case, or **Run all** at the right end of a group's title row to run that group's cases one after another. The top bar has **Open Scout app**, the only link between the two projects (the app itself never links to the simulator).
2. **Run page** (`/run/<id>`): an exact copy of the app's page (`ui/` is a byte-for-byte copy of `Place_Discover_Negotiating-AGENT/static/`), so you see what a real user sees. The fake user types letter by letter, reads the coordinator's questions, answers them, and is then checked. A small card at the bottom-left shows progress, Pause, Stop and the checks. In a Run all batch the next case opens by itself after 5 seconds (Cancel queue stops it).
3. **All runs** (`/runs`): every run, newest first, with totals (total, passed, failed, errored), filters, and an **Open** button per row. Open shows that run's saved chat in the app's page, read-only, with the coordinator's thinking and the final arguments (`/chat/<chat id>`). **Timeline** shows the same run as plain steps: who talked to whom, what each was given and what it returned, grouped by turn (`/timeline/<chat id>`).

4. **Evals** (`/evals`): one row per chat session, one compact result per eval. A card per eval at the top shows its counts and runs it on every session not yet run; press a result to open a side drawer with what was checked (the request, the judge's critique, the coordinator's memory). Press an eval's card or column title to focus the filters on that eval. Each eval says what it checks (Stated details: only the first scout call of the session; Helpful replies: every message the coordinator sent to the person, judged by the Gemini Decisions API for being helpful to what the person asked; Hand-off arguments: every scout call).

To add an eval to the page: write a function that turns its results into drawer items and a function that evaluates one session, then add one entry to `EVALS` in `server.py`. The page itself needs no change.

## Run it

Start the app first (it is the thing under test), then the simulator:

```powershell
cd ..\Place_Discover_Negotiating-AGENT ; .\.venv\Scripts\python -m app.server     # http://127.0.0.1:8000
cd ..\simulator ; .\start.ps1                                                       # http://127.0.0.1:8100
```

`APP_URL` points the simulator at another app instance (default `http://127.0.0.1:8000`). `SIM_PORT` changes the simulator's port.

## What is checked

For each run the simulator reads the app's event stream and checks: the scout agent's decisions in order (`ready`, or `needs_input` then `ready`), how many questions the coordinator asked, which details the first question was about, that the final result has a map pin (lat and lng), and that business, area and city are what the fake user said. A case passes only if every check holds. A model error (for example a rate limit) shows as **Errored**, not Failed.

## Files

| File | Purpose |
|---|---|
| `cases.json` | The 10 test cases: what the fake user types and what is expected. Edit it to add cases. |
| `server.py` | Serves the dashboard and run page, forwards `/api/chat` to the app, keeps the simulator's own history and results. |
| `sim.js` | The fake user (typing, answering, checking). Injected into the run page at serve time, so `ui/` stays an exact copy. |
| `dashboard.html`, `dash.js`, `dash.css` | The test-case dashboard (`dash.css` styles the All runs page too). |
| `runs.html`, `runs.js` | The All runs page. |
| `timeline.html`, `timeline.js`, `timeline.css` | The timeline page of a saved chat. |
| `replay.js` | Opens a saved chat read-only (injected into the app's page at `/chat/<id>`). |
| `ui/` | Copy of the app's page. After changing the app's UI, copy `static/*` here again. |
| `data/` | Simulator chats (`chats/`), every run (`runs.json`, newest 500) and each case's last result (`results.json`). Not the app's data. |

Each run uses the real app and the real model, so it spends model quota. The fake user's script is fixed, not model-written, so the same case is a fair repeat test.
