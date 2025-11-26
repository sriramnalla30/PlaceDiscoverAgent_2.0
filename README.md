# 🤖 Place Discovery AI Agent

An intelligent agentic system for discovering and analyzing local businesses using LLM-driven decision making, multi-stage reasoning, and transparent real-time workflow visualization.

## ✨ Key Features

- **🧠 LangGraph State Machine**: Multi-node workflow with conditional routing and checkpointed state
- **🤖 Multi-LLM Pipeline**: Groq (Llama-3.3-70b) for query parsing, routing, and sentiment analysis
- **🌐 Hybrid Web Scraping**: WebScraping.AI (residential proxies) + BeautifulSoup + Groq for intelligent review extraction
- **⚡ Real-time Streaming**: FastAPI SSE endpoints showing complete agent "thinking" process
- **📊 Interactive Dashboard**: Live visualization of tool usage, review analysis, and LLM decisions
- **🔍 Smart Search**: SerpStack API for location-based place discovery
- **🏆 Multi-Candidate Analysis**: Evaluates reviews from top 3 candidates before selecting winner

---

## 🎯 What It Does

**Input:** Natural language query (e.g., "Find the best gym in Koramangala, Bangalore")

**Agent Process:**

1. 🧭 Parses location & place type using LLM
2. 🔍 Searches SerpStack for top 10 results
3. 📊 Selects top 3 candidates by rating
4. 🌐 Scrapes Google reviews for all 3 (WebScraping.AI)
5. 🤖 LLM analyzes sentiment & quality of reviews
6. 🏆 Returns winner with detailed explanation

**Output:** Best place recommendation backed by review-based evidence and transparent reasoning

---

## 🏗️ Architecture

### Technology Stack

| Component     | Technology                     | Purpose                          |
| ------------- | ------------------------------ | -------------------------------- |
| **LLM**       | Groq (Llama-3.3-70b)           | Query parsing, routing, analysis |
| **Framework** | LangGraph                      | Stateful agent workflows         |
| **Backend**   | FastAPI                        | Async API with SSE streaming     |
| **Memory**    | MemorySaver                    | State checkpointing              |
| **Search**    | SerpStack API                  | Place discovery                  |
| **Scraping**  | WebScraping.AI + BeautifulSoup | HTML fetching & cleaning         |
| **Frontend**  | HTML/CSS/JS                    | Real-time dashboard              |

### Workflow Architecture

```
User Query → LLM Parser → Router Decision
                              ↓
                         PATH A: Review Analysis
                              ↓
              SerpStack Search (top 10 results)
                              ↓
              Select Top 3 by Rating/Reviews
                              ↓
              Review Extraction for ALL 3
              (WebScraping.AI → BS4 → Groq LLM)
                              ↓
              LLM Judge: Analyze Sentiment
                              ↓
              Winner Selection + Explanation
                              ↓
                            END
```

**PATH B (Negotiation)** - Placeholder for future development

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API Keys: Groq, SerpStack, WebScraping.AI

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Search_Agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_key
# GROQ_API_KEY_2=your_second_groq_key
# SERPSTACK_API_KEY=your_serpstack_key
# WEBSCRAPING_AI_API_KEY=your_webscraping_ai_key
```

### Run the Agent

```bash
# Start FastAPI server
python -m uvicorn app.main:app --reload

# Open dashboard in browser
# Navigate to: http://localhost:8000/dashboard.html
```

### Usage Example

1. Open `dashboard.html` in your browser
2. Enter query: "Find the best gym in Koramangala, Bangalore"
3. Watch real-time agent thinking:
   - Tool usage badges (🔍 SerpStack, 🤖 WebScraping.AI)
   - Review content for all 3 candidates
   - LLM decision process
4. Get winner recommendation with explanation

---

## 📁 Project Structure

```
Search_Agent/
├── app/
│   ├── main.py              # FastAPI endpoints & SSE streaming
│   ├── config.py            # Environment configuration
│   ├── models.py            # Pydantic data models
│   └── agent/
│       ├── graph.py         # LangGraph workflow definition
│       ├── nodes.py         # Agent nodes (revisor, analyzer, etc.)
│       ├── state.py         # Shared state schema
│       └── tools.py         # External API tools (SerpStack, WebScraping.AI)
├── data/                    # Checkpoints storage (auto-created)
├── dashboard.html           # Interactive frontend UI
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── start_server.bat         # Windows batch script to start server
└── README.md                # This file
```

---

## 🔑 Required API Keys

Add these to your `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_2=your_second_groq_key
SERPSTACK_API_KEY=your_serpstack_key
WEBSCRAPING_AI_API_KEY=your_webscraping_ai_key
```

**Get API Keys:**

- [Groq Console](https://console.groq.com/) - Free tier available
- [SerpStack](https://serpstack.com/) - 100 free searches/month
- [WebScraping.AI](https://webscraping.ai/) - 1000 free requests/month

---

## 🎯 Implementation Details

### PATH A: Review-Based Selection (✅ Implemented)

**Nodes:**

1. **Revisor Node** - Routes queries to appropriate path (A or B)
2. **Simple Best Reviewed Node** - Fetches top 10 from SerpStack, selects top 3 by rating
3. **Review Extraction Node** - Scrapes Google reviews for all 3 candidates
4. **Analyze Reviews Node** - LLM evaluates sentiment, picks winner with explanation

**Key Technologies:**

- **Structured Outputs**: Uses Pydantic models for LLM responses (type safety)
- **Hybrid Scraping**: WebScraping.AI HTML → BeautifulSoup cleaning → Groq extraction
- **State Checkpointing**: MemorySaver persists workflow state across streaming sessions
- **SSE Streaming**: Frontend receives real-time events (node_start, tool_usage, llm_response)

### Dashboard Features

**Real-time Visualization:**

- 🔧 Tool usage badges (SerpStack, WebScraping.AI + Groq)
- 📝 Review content display for all candidates
- 🎯 LLM decision with reasoning
- ⏱️ Node execution tracking
- 🔄 Streaming event handling

**Event Types:**

- `data_fetched` - SerpStack results
- `reviews_fetched` - Extracted reviews
- `node_start` - Agent node execution
- `llm_response` - LLM decision output

---

## 🛠️ Development

### Testing the Review Tool

```bash
# Test WebScraping.AI + Groq integration
python test_review_tool.py
```

### Key Code Locations

**Agent Logic:**

- `app/agent/nodes.py` - Workflow nodes
- `app/agent/graph.py` - LangGraph state machine
- `app/agent/tools.py` - API integrations

**Streaming:**

- `app/main.py` - `/agent/start` (state init) & `/stream` (SSE execution)

**Frontend:**

- `dashboard.html` - SSE client with event handlers

---

## 🔮 Future Enhancements (PATH B)

**Planned Features:**

- 📞 Phone number extraction from search results
- 🔍 Tavily API deep research
- 💰 Pricing strategy generation
- 👤 Human-in-the-loop approval panel
- 🤝 Simulated negotiation workflow

---

## 📊 Performance Notes

- **Groq API**: ~1-2s per LLM call (Llama-3.3-70b)
- **SerpStack**: ~1-3s per search
- **WebScraping.AI**: ~3-5s per page (residential proxies)
- **Total Workflow**: ~15-25s for complete analysis

---

## 👤 Author

Built as an AI agentic system demonstration project showcasing:

- LangGraph state machines
- Multi-stage LLM reasoning
- Hybrid web scraping techniques
- Real-time streaming architectures

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- PATH B negotiation workflow implementation
- Additional scraping fallback strategies
- Enhanced error handling
- Performance optimization
- UI/UX improvements

---

## 🐛 Troubleshooting

**Issue**: Agent not streaming in real-time  
**Solution**: Ensure `/agent/start` only saves state (doesn't execute), streaming happens in `/stream`

**Issue**: Review extraction failing  
**Solution**: Check WebScraping.AI API quota, verify API key in `.env`

**Issue**: No results from SerpStack  
**Solution**: Validate API key, check query format (needs location + place type)

---

**Last Updated**: November 2025

Input:
<img width="1919" height="1076" alt="image" src="https://github.com/user-attachments/assets/0a6edb15-c93d-4f43-a18e-974b04cc39d0" />

Procesing:
<img width="1919" height="1064" alt="image" src="https://github.com/user-attachments/assets/326e7c91-281f-4aaa-a921-28068f223458" />
<img width="1027" height="936" alt="image" src="https://github.com/user-attachments/assets/4bd7b80b-9183-4a35-b63b-9de4388b4076" />
<img width="1047" height="952" alt="image" src="https://github.com/user-attachments/assets/9b132ea4-22f3-4200-a6fe-e9d35f71525f" />
<img width="1194" height="947" alt="image" src="https://github.com/user-attachments/assets/5dfccf21-6433-4fd9-a917-d76d0cd59258" />
<img width="713" height="932" alt="image" src="https://github.com/user-attachments/assets/b614cc7c-9d47-44d0-badc-ec4f2582105c" />


