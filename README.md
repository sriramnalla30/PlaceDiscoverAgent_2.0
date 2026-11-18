# PlaceDiscoverAgent 2.0

A place-finding assistant for people in India. A **coordinator** agent talks to the person and keeps control of the flow. A **scout** agent turns the request into a Google Maps search (it finds the map pin, runs the scraper) and the page shows every place with all its data: rating, reviews, opening hours (open or closed right now), distance, phone, links, and everything else Google returned. The project also holds the evals and the simulator that test the agents.

**Live:** https://ramplacesearch.netlify.app

---

## Older versions

The earlier versions of this project (SerpStack search, review analysis and SMS negotiation) are kept on the branch [`older-version`](https://github.com/sriramnalla30/PlaceDiscoverAgent_2.0/tree/older-version) and in this repository's history.

**Video Demo:** https://drive.google.com/file/d/1HZJYOx-oEaGraIHzRkI5JS-NOYnM-SDk/view?usp=drive_link

**Concepts Explained:** https://drive.google.com/file/d/1piOMcCW5CnnaE6LSVltsSZVLI1VurMfT/view?usp=drive_link

---

## What is in this repository

| Folder | What it is |
|---|---|
| `Place_Discover_Negotiating-AGENT/` | The app: coordinator and scout agents (LangGraph, Groq), the Google Maps search, the chat page, deployment files. Its own `README.md` explains it. |
| `simulator/` | A fake user that drives the app, a dashboard of test cases, run pages and the Evals page. |
| `evals suit/` | The three evals: stated details (scout agent), helpful replies (coordinator), hand-off arguments. `evals.md` describes each. |
| `SAMPLE_TEST_DASHBOARDS/` | The Maps Scraper Lab, a small dashboard for trying the Google Maps scraper by hand. |
| `documentary/` | Notes on the problems met while building it. |

The folders sit side by side on purpose: the simulator and the evals find the app's `.env` and virtual environment in `../Place_Discover_Negotiating-AGENT`.

## Run it

```powershell
cd Place_Discover_Negotiating-AGENT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
copy .env.example .env        # then add GROQ_API_KEY
.\.venv\Scripts\python.exe -m app.server      # http://127.0.0.1:8000
```

The Google Maps scraper binary (`gmaps`, gosom google-maps-scraper v1.18.1) goes in `Place_Discover_Negotiating-AGENT/bin/`, or set `GMAPS_BIN`. `Place_Discover_Negotiating-AGENT/run.txt` lists every server (app, simulator, scraper lab) and how to start each.

## Deployment

The page runs on Netlify and the API on Render (a Docker image with Chromium and the scraper). See `Place_Discover_Negotiating-AGENT/docs/DEPLOYMENT.md`.

## Tests

```powershell
cd Place_Discover_Negotiating-AGENT
.\.venv\Scripts\python.exe -m unittest discover -s tests -v     # offline: no model, no network
```

The evals need Langfuse and Gemini keys in the app's `.env`; see `evals suit/evals.md`.

---

**Repository:** [github.com/sriramnalla30/PlaceDiscoverAgent_2.0](https://github.com/sriramnalla30/PlaceDiscoverAgent_2.0)
