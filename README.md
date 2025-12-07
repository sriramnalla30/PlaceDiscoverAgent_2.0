# PlaceDiscoverAgent 2.0

A local business discovery and negotiation agent built with LangGraph and FastAPI. This system combines web search, review analysis, and SMS-based negotiation with human approval at every step.

---

## Overview

This agent helps you find local businesses, analyze their reviews, and negotiate deals via SMS. The workflow uses a reflection pattern where the system drafts messages, you review and approve them, and the agent learns from responses to continue the conversation.

**Core Features:**

- Natural language business search powered by SerpStack
- Review scraping and sentiment analysis from Google Maps
- SMS negotiation with shops using their real phone numbers
- Human-in-the-loop approval before every message
- Real-time streaming dashboard showing agent reasoning

---

## How Reflection Works

**Reflection Pattern:**  
Generate → Critique → Revise → Decide to continue or end

**Applied in this agent:**

1. **Responder** drafts a message using context (reviews, pricing, user goals)
2. **Revisor** critiques the draft for tone, leverage, and strategy
3. **Human approval** required before sending (you can edit the message)
4. **Loop continues** based on shop replies until deal reached or declined

**Visual References:**

- `reflection_agent.png` - Basic reflection loop concept
- `reflexion_agent.png` - Advanced reflexion pattern with tool use
- `data_flow.png` - Complete system data flow

---

## Complete User Journey

1. **Search Query**  
   Type: _"Find the best gym in Koramangala, Bangalore and negotiate membership"_

2. **Intent Parsing**  
   LLM extracts: City (Bangalore), Area (Koramangala), Type (Gym), Intent (Negotiate)

3. **Business Discovery**  
   SerpStack fetches ~10 local businesses with:

   - Name, address, phone number
   - Rating and review count
   - Basic business info

4. **Review Analysis**

   - Top 3 businesses selected by rating
   - WebScraping.AI extracts actual Google Maps reviews
   - LLM analyzes sentiment, quality, common complaints/praises
   - Best option selected with reasoning

5. **Results Display**  
   Dashboard shows:

   - **Recommended business** (highlighted green card)
   - All results in grid with quick actions
   - Maps button (opens Google Maps to location)
   - Call button (click-to-call on mobile)
   - Negotiate button (starts SMS workflow)

6. **Negotiation Setup**  
   Click Negotiate → Enter:
   - Your goal (e.g., "Get monthly membership under ₹3000")
   - Target price (optional)
7. **Message Drafting**  
   Agent creates draft message using:
   - Review insights for leverage
   - Your stated goal
   - Professional, friendly tone
8. **Human Approval**  
   Yellow sticky note panel shows:

   - Proposed message
   - Agent's strategy/reasoning
   - Edit box (modify if needed)
   - Approve or Cancel buttons

9. **SMS Exchange**

   - Message sent to shop's phone (from SerpStack)
   - Chat interface opens on right side
   - Auto-polls for replies every 10 seconds
   - Each reply analyzed by agent
   - New draft suggested (back to step 8)

10. **Resolution**  
    Continues until:
    - Shop agrees to terms
    - Shop declines
    - You manually end negotiation

---

## System Architecture

### Technology Stack

| Layer             | Technology              | Purpose                                                |
| ----------------- | ----------------------- | ------------------------------------------------------ |
| **LLM**           | Groq (Llama 3.3 70B)    | Intent parsing, message drafting, sentiment analysis   |
| **Orchestration** | LangGraph               | State machine with conditional routing and checkpoints |
| **Backend**       | FastAPI                 | REST API with Server-Sent Events for real-time updates |
| **Search**        | SerpStack API           | Local business discovery with contact details          |
| **Scraping**      | WebScraping.AI          | Google Maps review extraction                          |
| **SMS**           | SMSMobileAPI            | Send/receive SMS (works via Android app)               |
| **Frontend**      | Vanilla HTML/CSS/JS     | Dashboard with live streaming and chat interface       |
| **State Storage** | MemorySaver (in-memory) | Conversation state persistence                         |

### LangGraph Workflow

```
START
  ↓
REVISOR (Router Node)
  ↓
  ├──> PATH A: Simple Search
  │      ├─ Fetch from SerpStack
  │      ├─ Select top 3 by rating
  │      ├─ Scrape reviews
  │      ├─ Analyze with LLM
  │      └─ Return best option → END
  │
  └──> PATH B: Negotiation
         ├─ Init negotiation state
         ├─ Strategy formulation
         ├─ HUMAN REVIEW (⏸️ INTERRUPT)
         ├─ Send SMS
         ├─ Poll for reply
         ├─ Analyze response
         └─ Loop back to Strategy (or END if resolved)
```

**Key Features:**

- **Conditional routing** based on user intent
- **Interrupts** pause execution for human approval
- **Streaming** pushes real-time events to frontend
- **State persistence** maintains context across sessions

---

## Project Structure

```
PlaceDiscoverAgent_2.0/
│
├── app/
│   ├── main.py                 # FastAPI routes, SSE endpoints
│   ├── config.py               # Environment variable config
│   ├── models.py               # Pydantic request/response schemas
│   │
│   ├── agent/
│   │   ├── graph.py            # LangGraph workflow definition
│   │   ├── nodes.py            # All workflow nodes (responder, revisor, etc.)
│   │   ├── state.py            # Shared state schema
│   │   └── tools.py            # SerpStack, WebScraping.AI integrations
│   │
│   ├── messaging/
│   │   ├── base.py             # Messaging provider interface
│   │   ├── service.py          # Provider factory
│   │   └── smsmobileapi.py     # SMSMobileAPI implementation
│   │
│   └── static/
│       └── style.css           # Dashboard styles
│
├── data/
│   └── checkpoints.db          # Runtime state (gitignored)
│
├── backend/
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   └── dashboard.html          # Main UI (served at localhost:8000/frontend/dashboard.html)
│
├── reflection_agent.png        # Concept diagram
├── reflexion_agent.png         # Advanced pattern diagram
├── data_flow.png               # System flow diagram
│
├── .env                        # Your API keys (NOT COMMITTED)
├── .env.example                # Template for .env
├── .gitignore                  # Excludes .env, checkpoints.db
└── README.md                   # This file
```

---

## Detailed Setup Guide

### Prerequisites

1. **Python 3.10 or higher**  
   Check: `python --version`

2. **API Keys** (all have free tiers):

   - **Groq**: [console.groq.com](https://console.groq.com/) → Create account → Copy API key
   - **SerpStack**: [serpstack.com](https://serpstack.com/) → Sign up → Get free API key (100 searches/month)
   - **WebScraping.AI**: [webscraping.ai](https://webscraping.ai/) → Sign up → Get API key (1000 requests/month)
   - **SMSMobileAPI**: [smsmobileapi.com](https://www.smsmobileapi.com/) → Install Android app → Generate API key

3. **Phone for SMS** (if using SMSMobileAPI):
   - Android phone with active SIM
   - Install SMSMobileAPI app from Play Store
   - Keep phone connected to internet

---

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/sriramnalla30/PlaceDiscoverAgent_2.0.git
cd PlaceDiscoverAgent_2.0
```

#### 2. Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` prefix in your terminal.

#### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

**What gets installed:**

- `fastapi` + `uvicorn` — Web framework and server
- `langchain` + `langgraph` — LLM orchestration
- `groq` — Groq API client
- `requests` — HTTP requests
- `beautifulsoup4` — HTML parsing
- `pydantic-settings` — Config management
- `smsmobileapi` — SMS provider (if using SMSMobileAPI)

#### 4. Configure Environment Variables

Create a file named `.env` in the project root:

```bash
# === LLM Configuration ===
GROQ_API_KEY=gsk_your_primary_groq_key_here
GROQ_API_KEY_2=gsk_optional_backup_key_here

# === Search & Scraping ===
SERPSTACK_API_KEY=your_serpstack_api_key
WEBSCRAPING_AI_API_KEY=your_webscraping_ai_key

# === SMS Provider ===
messaging_provider=smsmobileapi
smsmobileapi_key=your_smsmobileapi_key_here

# === Phone Number Settings ===
# Leave BLANK to use shop's phone from SerpStack
# Only set if you need fallback when shop has no phone
default_target_number=
default_sender_number=

# === LangSmith (Optional Debugging) ===
langchain_tracing_v2=true
langchain_endpoint=https://api.smith.langchain.com
langchain_api_key=your_langsmith_key_optional
langchain_project=PlaceDiscoverAgent
```

**Important Notes:**

- **Never commit `.env`** — Already in `.gitignore`
- **`default_target_number`** — Leave blank unless you want to test without real shop phones
- **Two Groq keys** — Recommended to avoid rate limits
- **SMSMobileAPI key** — Get from app settings after installing on your Android phone

#### 5. Verify Setup

Check if all dependencies installed correctly:

```bash
python -c "import fastapi, langchain, groq; print('All imports successful!')"
```

Should print: `All imports successful!`

#### 6. Start the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX]
INFO:     Started server process [XXXXX]
INFO:     Application startup complete.
```

#### 7. Access Dashboard

Open browser and go to:

```
http://localhost:8000/frontend/dashboard.html
```

You should see:

- Header: "🔍 Place Discover"
- Search card with textarea
- Step indicator (1-6)
- SMS gateway notice

---

### Testing the System

#### Quick Test (No SMS)

1. Enter query: `"Find the best cafe in Indiranagar, Bangalore"`
2. Click **Start Scouting**
3. Watch:

   - Steps light up (1 → 2 → 3 → 4 → 5)
   - Logs show tool usage
   - SerpStack results displayed
   - Reviews analyzed
   - Best recommendation highlighted

4. Check results:
   - Green card shows best pick
   - All options in grid below
   - Maps/Call buttons work
   - **Don't click Negotiate yet** (requires SMS setup)

#### Full Test (With SMS)

1. **Prerequisites:**

   - SMSMobileAPI app installed and running
   - API key in `.env`
   - Phone has internet and SMS

2. Enter query: `"Find gyms in Koramangala and negotiate monthly fee"`

3. Click best result's **Negotiate** button

4. Fill modal:

   - Goal: "Get monthly membership under ₹2500"
   - Target Price: 2500

5. Click **Launch Agent**

6. **HITL Panel appears** (yellow sticky note):

   - Review proposed message
   - Edit if needed
   - Click **Approve & Contact**

7. **Chat opens** on right side:
   - Message sent to shop
   - Wait for reply (polls every 10 seconds)
   - Agent suggests response
   - Approve again
   - Continue loop

---

## Production Deployment

### Environment Variables

**DO NOT hardcode:**

- API keys
- Phone numbers
- Passwords

**Use platform's secret management:**

- Railway → Settings → Variables
- Render → Environment → Secret Files
- Fly.io → Secrets
- Docker → .env file (not committed)

### Example Production `.env`

```bash
# Minimal production config
GROQ_API_KEY=${GROQ_KEY}  # Injected by platform
SERPSTACK_API_KEY=${SERP_KEY}
WEBSCRAPING_AI_API_KEY=${SCRAPE_KEY}
messaging_provider=smsmobileapi
smsmobileapi_key=${SMS_KEY}
environment=production
```

### Run Command (Production)

```bash
# Don't use --reload in production
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
```

### Security Checklist

- ✅ `.env` in `.gitignore`
- ✅ CORS configured for your domain only
- ✅ API keys rotated regularly
- ✅ Rate limiting enabled (if high traffic)
- ✅ HTTPS enabled (use reverse proxy)
- ✅ Phone with SMSMobileAPI always connected

### Monitoring

- **LangSmith** — Trace LLM calls and debug workflows
- **Application logs** — Check `uvicorn` output for errors
- **API quotas** — Monitor Groq, SerpStack, WebScraping.AI usage

---

## Code Deep Dive

### Key Files Explained

#### `app/agent/graph.py`

Defines the LangGraph workflow:

- Creates `StateGraph` with `AgentState` schema
- Adds nodes (revisor, strategy, human_review, etc.)
- Sets conditional edges (PATH A vs PATH B)
- Configures interrupts (pause at HITL)
- Compiles with MemorySaver checkpoint

#### `app/agent/nodes.py`

Implements all workflow nodes:

- `revisor_node` — Routes to PATH A or B
- `simple_best_reviewed_node` — Sorts by rating
- `review_extraction_node` — Scrapes Google Maps
- `analyze_reviews_node` — LLM sentiment analysis
- `negotiation_path_node` — Initializes SMS workflow
- `strategy_node` — Drafts message
- `human_review_node` — Sends SMS after approval
- `negotiation_manager_node` — Polls replies, decides to continue/end

#### `app/agent/tools.py`

External API integrations:

- `search_places(query)` — Calls SerpStack
- `extract_reviews(place_name)` — WebScraping.AI + BeautifulSoup + LLM

#### `app/messaging/smsmobileapi.py`

SMS provider implementation:

- `send_message(to, message)` — Sends SMS via API
- `get_messages()` — Fetches inbox messages

#### `app/main.py`

FastAPI application:

- `/agent/start` — Initialize workflow, save state
- `/agent/stream` — SSE endpoint for real-time updates
- `/agent/negotiate/start` — Begin negotiation for a place
- `/agent/approve` — Resume graph after HITL approval
- `/agent/check-reply` — Poll for SMS replies
- `/agent/send-chat` — Send continuation message
- `/agent/terminate` — End negotiation

#### `frontend/dashboard.html`

Frontend with:

- Search form and step indicator
- SSE client (connects to `/agent/stream`)
- Log container for real-time events
- Results grid with Maps/Call/Negotiate buttons
- HITL approval panel
- Chat interface for SMS negotiation

---

## Troubleshooting

### Issue: Server won't start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`  
**Fix:**

```bash
# Make sure venv is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r backend/requirements.txt
```

### Issue: "No API key configured"

**Error:** `APIKeyError: GROQ_API_KEY not set`  
**Fix:**

- Check `.env` file exists in project root
- Verify key format: `GROQ_API_KEY=gsk_...`
- Restart server after editing `.env`

### Issue: SerpStack returns no results

**Possible causes:**

- Invalid API key
- Query doesn't include location
- API quota exhausted

**Fix:**

- Test key at [serpstack.com/dashboard](https://serpstack.com/dashboard)
- Include city in query: "gyms in Bangalore"
- Check quota limits

### Issue: SMS not sending

**Checklist:**

- SMSMobileAPI app running on phone
- Phone has internet connection
- Phone has active SIM with SMS capability
- API key correct in `.env`
- `messaging_provider=smsmobileapi` set

**Debug:**

```bash
# Check logs for SMS errors
python -m uvicorn app.main:app --reload --log-level debug
```

### Issue: Frontend shows blank page

**Fix:**

- Check URL: `http://localhost:8000/frontend/dashboard.html` (with `.html`)
- Check browser console for errors (F12)
- Verify static files mounted correctly in `app/main.py`

### Issue: HITL panel doesn't appear

**Cause:** Graph not paused correctly  
**Fix:**

- Verify `interrupt_before=["human_review"]` in `graph.py`
- Check state update after `/agent/negotiate/start`

---

## Contributing

Contributions welcome! Areas for improvement:

1. **Multi-provider SMS** — Add Twilio, Nexmo support
2. **Voice calls** — Add phone call capability
3. **Better review analysis** — More sophisticated sentiment models
4. **Caching** — Redis for search results
5. **UI enhancements** — Mobile-responsive design
6. **Analytics** — Track negotiation success rates
7. **Testing** — Unit and integration tests

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Repository:** [github.com/sriramnalla30/PlaceDiscoverAgent_2.0](https://github.com/sriramnalla30/PlaceDiscoverAgent_2.0)

**Issues:** Use GitHub Issues for bugs and feature requests

---

## Acknowledgments

- LangGraph team for the state graph framework
- Groq for fast LLM inference
- SerpStack for local business search
- WebScraping.AI for reliable scraping
- SMSMobileAPI for SMS gateway

---

**Last Updated:** December 2025
