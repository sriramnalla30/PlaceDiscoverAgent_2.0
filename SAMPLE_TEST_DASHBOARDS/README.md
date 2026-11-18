# Maps Scraper Lab

Local dashboard for the [gosom google-maps-scraper](https://github.com/gosom/google-maps-scraper) in **fast mode**. It replaces the earlier Serpstack lab. Python 3.10+ standard library only; no packages, no API key, no LLM.

```powershell
cd D:\AI_Projects\Search_Agent_2\SAMPLE_TEST_DASHBOARDS
python server.py          # or .\start.ps1
```

Open **http://127.0.0.1:8765**. Use `python server.py --port 8766` if the port is busy. Restart the server after pulling code changes.

## What it needs

The scraper binary at `..\test_check\bin\gmaps.exe` (v1.18.1 Windows build). Set `GMAPS_BIN` to use another path. On first use the scraper may download a headless Chromium into your user profile; fast mode itself starts no browser.

## How to use

1. Pick a business type and area, or type a list query such as `gyms in Koramangala Bangalore`.
2. Check the location pin (latitude/longitude). Fast mode refuses to run without one.
3. Optionally set a radius (100-50000 m) and language.
4. **Run scrape.** One click is one live scrape; only one runs at a time.

Tabs: **Places** (table, click a row for every field, map of results), **Phone check** (kind, +91 form, shared numbers), **Attributes** (fill rate of every field), **Raw JSON**, **Saved runs** (restores inputs and output). Export to CSV or JSON.

The **maps_graph** button (top right) explains the flow, the inputs, and every output attribute with measured coverage.

## Things to know

- Fast mode returns at most **20 places** per query and only for **list searches**. A query that names one business returns nothing.
- `-depth` and `-zoom` have no effect in fast mode, so the dashboard does not expose them.
- Fast mode is beta. The scraper's normal mode failed on all test jobs on 2026-10-30.
- `_phone_kind`, `_phone_e164` and `_phone_shared` are added by this dashboard. The kind is a guess from the number's prefix.
- Scraping Google may violate its terms of service (see the scraper's README).
- Saved runs live in `history/maps_runs.sqlite3` (excluded from Git). The old Serpstack runs, if any, are in `history/runs.sqlite3` and are no longer used.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests mock the scraper process; they make no Google requests.
