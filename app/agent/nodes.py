"""
YOUR DESIGN: STEP 5 Revisor Node Implementation

STEP 5 Logic:
- Receives: user_query + LLM parsed params + SERPSTACK results (from checkpoint memory)
- Uses: Gemini LLM to decide PATH A or PATH B
- PATH A: User wants best reviewed → Sort by rating/reviews → Return top result
- PATH B: User wants negotiation/pricing → Use Tavily API → Deep search → Negotiation workflow
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agent.state import AgentState
from app.config import settings
from pydantic import BaseModel, Field
from typing import Literal
import logging
import json

logger = logging.getLogger(__name__)


# ==========================================
# HELPER: Get Groq LLM
# ==========================================

def get_llm(temperature: float = 0.3, use_key_2: bool = False):
    """
    Get configured Groq LLM
    
    Args:
        temperature: LLM temperature (0-1)
        use_key_2: If True, uses GROQ_API_KEY_2, else uses GROQ_API_KEY
    """
    api_key = settings.groq_api_key_2 if use_key_2 else settings.groq_api_key
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=api_key or settings.groq_api_key  # Fallback to main key
    )


# ==========================================
# STEP 5: REVISOR NODE (Decision Point)
# ==========================================

class PathDecision(BaseModel):
    """Gemini's decision on which path to follow"""
    path: Literal["path_a", "path_b"] = Field(
        description="path_a for simple best reviewed, path_b for negotiation"
    )
    show_all: bool = Field(
        description="True (boolean) if user wants to see ALL results, False (boolean) if only best/top one. Do not return a string."
    )
    reasoning: str = Field(
        description="Why this path was chosen"
    )
    confidence: float = Field(
        description="Confidence score 0.0 to 1.0 (number). Do not return a string."
    )


async def revisor_node(state: AgentState) -> dict:
    """
    STEP 5: Revisor Node - Gemini decides PATH A or PATH B
    
    Analyzes:
    - user_query (original natural language)
    - user_intent (parsed by LLM in STEP 2)
    - serp_results (SERPSTACK data from STEP 4)
    
    Decision Logic:
    PATH A: User asks for "best reviewed", "top rated", "highest rating"
    PATH B: User mentions "negotiate", "price", "budget", "fees", "cost"
    """
    
    logger.info("🔍 STEP 5: Revisor Node - Analyzing query intent...")
    
    # Get data from checkpoint memory
    user_query = state.get("user_query", "")
    user_intent = state.get("user_intent", "")
    serp_results = state.get("serp_results", [])
    parsed_params = state.get("parsed_params", {})
    
    llm = get_llm(temperature=0.3, use_key_2=True)  # Use API_KEY_2 for Revisor
    structured_llm = llm.with_structured_output(PathDecision)
    
    system_prompt = """You are a decision-making agent for a place discovery system.

Your job: Decide TWO things:
1. PATH: path_a (simple search) or path_b (negotiation/pricing)
2. SHOW_ALL: True (show all results) or False (show only best one)

IMPORTANT: Return raw JSON booleans (true/false) and numbers (0.8), NOT strings ("true"/"0.8").

PATH A Indicators:
- "best reviewed", "top rated", "highest rating", "most popular", "show me the best"
- Generic queries like "gyms in X", "restaurants in Y" (without "best")

PATH B Indicators:
- "negotiate", "price", "pricing", "cost", "fees"
- "budget", "cheap", "affordable"
- "contact", "call", "phone number"
- "deals", "discount"

SHOW_ALL Rules:
- show_all = True if user says: "gyms", "restaurants", "show all", "list", "display other", "2nd and 3rd positions", "top 3", "top 5", "top 10", "multiple"
- show_all = False if user says: "best", "top one", "the highest", "most reviewed" (UNLESS they specify a number > 1 like "top 3")

Examples:
- "gyms in Bangalore" → path_a, show_all=True (user wants to see ALL gyms)
- "best gym in Bangalore" → path_a, show_all=False (user wants THE BEST one)
- "top 3 gyms in Bangalore" → path_a, show_all=True (user wants a LIST)
- "gyms with pricing in Bangalore" → path_b, show_all=True (negotiation needed for all)
"""
    
    user_prompt = f"""
Original User Query: "{user_query}"
Parsed Intent: "{user_intent}"
Number of SERPSTACK Results: {len(serp_results)}

Sample Result (if available):
{serp_results[0] if serp_results else "No results"}

Decision: Which path should we follow?
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    decision: PathDecision = structured_llm.invoke(messages)
    
    logger.info(f"✅ Revisor Decision: {decision.path.upper()} | Show All: {decision.show_all}")
    logger.info(f"💭 Reasoning: {decision.reasoning}")
    logger.info(f"📊 Confidence: {decision.confidence:.2f}")
    
    return {
        "route": decision.path,  # Store decision in state
        "show_all": decision.show_all,  # NEW: Store whether to show all results
        "current_step": "revisor",
        "messages": state["messages"] + [
            AIMessage(content=f"🤖 Revisor Analysis:\n- Path: {decision.path.upper()}\n- Show All Results: {decision.show_all}\n- Reasoning: {decision.reasoning}\n- Confidence: {decision.confidence:.0%}")
        ]
    }


# ==========================================
# DECISION FUNCTION (for conditional edge)
# ==========================================

def decide_path(state: AgentState) -> Literal["path_a", "path_b"]:
    """
    Conditional edge function: Returns which path to follow
    """
    route = state.get("route", "path_a")
    logger.info(f"📍 Routing to: {route}")
    return route


# ==========================================
# PATH A: SIMPLE BEST REVIEWED
# ==========================================

async def simple_best_reviewed_node(state: AgentState) -> dict:
    """
    PATH A: Simple Best Reviewed Analysis
    
    Process:
    1. Get SERPSTACK results from state
    2. Check if user wants ALL results or just BEST one (from revisor decision)
    3. If show_all=True: Return ALL sorted by rating
    4. If show_all=False: Use Tavily for deep analysis, return top 1
    5. END workflow
    """
    
    logger.info("⭐ PATH A: Analyzing places...")
    
    serp_results = state.get("serp_results", [])
    show_all = state.get("show_all", False)  # Get revisor's decision
    
    if not serp_results:
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": True,
            "recommendations": [],
            "messages": state["messages"] + [
                AIMessage(content="❌ No places found in SERPSTACK results.")
            ]
        }
    
    # Sort by rating (primary) and reviews_count (secondary)
    sorted_places = sorted(
        serp_results,
        key=lambda x: (x.get("rating", 0), x.get("reviews_count", 0)),
        reverse=True
    )
    
    # DECISION POINT: Show all or just best?
    # Force show_all if user explicitly asked for "top X" or "list" in query
    user_query_lower = state.get("user_query", "").lower()
    logger.info(f"DEBUG: user_query='{user_query_lower}'")
    force_show_all = any(x in user_query_lower for x in ["top 3", "top 5", "top 10", "list", "all", "multiple"])
    logger.info(f"DEBUG: show_all={show_all}, force_show_all={force_show_all}")
    
    if show_all or force_show_all:
        logger.info(f"📋 Showing ALL {len(sorted_places)} results (user wants complete list)")
        
        # Return all results sorted by rating
        result_text = f"✅ **ALL PLACES IN YOUR AREA** (Sorted by Rating):\n\n"
        
        for idx, place in enumerate(sorted_places, 1):
            reviews_count = place.get('reviews_count', 0)
            # Safely convert to int for formatting
            try:
                reviews_count = int(reviews_count) if reviews_count else 0
            except (ValueError, TypeError):
                reviews_count = 0
            
            result_text += f"""
**{idx}. {place.get('name', 'Unknown')}**
⭐ Rating: {place.get('rating', 'N/A')}/5.0 ({reviews_count:,} reviews)
📍 Address: {place.get('address', 'N/A')}
📞 Phone: {place.get('phone', 'Not available')}
---
"""
        
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": True,
            "recommendations": sorted_places,  # Return all
            "messages": state["messages"] + [
                AIMessage(content=result_text)
            ]
        }
    
    else:
        logger.info(f"🏆 Finding THE BEST option (user wants top recommendation)")
        
        # Get top 3 candidates for deep analysis
        top_candidates = sorted_places[:min(3, len(sorted_places))]
        
        logger.info(f"📊 Top candidates for deep analysis: {[p.get('name') for p in top_candidates]}")
        
        # We return these 3 candidates so the next node (review_extraction) can fetch reviews for ALL of them
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": False, # Not complete yet
            "recommendations": top_candidates,  # Return TOP 3
            "messages": state["messages"] + [
                AIMessage(content=f"Found {len(top_candidates)} top candidates. Fetching detailed reviews to pick the winner...")
            ]
        }


# ==========================================
# PATH B: NEGOTIATION WORKFLOW (Placeholder)
# ==========================================

async def negotiation_path_node(state: AgentState) -> dict:
    """
    PATH B: Negotiation Workflow (Under Construction)
    
    Future Implementation (STEP 6-9):
    1. Extract phone numbers from SERPSTACK
    2. Call Tavily API for deep research:
       - Negotiation skills
       - How others work at that place
       - Minimum budget knowledge
       - Pricing insights
    3. Prepare negotiation strategy
    4. HITL: User interacts with negotiation options
    5. Final recommendation with pricing
    
    For now: Returns placeholder message
    """
    
    logger.info("🚧 PATH B: Negotiation workflow (Under construction)")
    
    serp_results = state.get("serp_results", [])
    user_intent = state.get("user_intent", "")
    
    # Extract phone numbers from SERPSTACK results
    places_with_phones = [
        place for place in serp_results 
        if place.get("phone") and place.get("phone") != ""
    ]
    
    placeholder_text = f"""
🚧 **NEGOTIATION WORKFLOW - UNDER CONSTRUCTION**

Your request requires negotiation/pricing information.

**Current Status:**
- ✅ Found {len(serp_results)} places from SERPSTACK
- ✅ {len(places_with_phones)} places have phone numbers
- 🚧 Tavily API deep search - Coming soon!
- 🚧 Negotiation strategy analysis - Coming soon!
- 🚧 HITL interaction panel - Coming soon!

**Next Steps (Will be implemented):**
1. Use Tavily API to research:
   - Negotiation skills for this type of place
   - How others work at these locations
   - Minimum budget insights
   - Pricing patterns
2. Prepare personalized negotiation strategy
3. Show interactive options for user approval

**For now:** Check back after PATH B implementation!
"""
    
    return {
        "current_step": "negotiation_path",
        "is_complete": True,
        "recommendations": {
            "status": "under_construction",
            "places_count": len(serp_results),
            "places_with_phones": len(places_with_phones)
        },
        "messages": state["messages"] + [
            AIMessage(content=placeholder_text)
        ]
    }


# ==========================================
# STEP 6: REVIEW EXTRACTION NODE
# ==========================================

async def review_extraction_node(state: AgentState) -> dict:
    """
    STEP 6: Review Extraction Node
    
    Fetches detailed reviews for the recommended places using WebScraping.AI.
    """
    logger.info("🔍 STEP 6: Review Extraction Node")
    
    recommendations = state.get("recommendations", [])
    if not recommendations:
        logger.info("No recommendations to fetch reviews for.")
        return {}
        
    from app.agent.tools import fetch_reviews_webscraping_ai
    
    reviews_map = {}
    updated_recommendations = []
    
    for place in recommendations:
        name = place.get("name", "Unknown")
        address = place.get("address", "")
        
        # Construct query for WebScraping.AI
        query = f"{name} {address}".strip()
        
        logger.info(f"Fetching reviews for: {name}")
        
        # Call the tool directly (it returns a list of strings)
        # Note: In a real async environment, we might want to run these in parallel
        try:
            # The tool is decorated, so we invoke it
            reviews = fetch_reviews_webscraping_ai.invoke({"query": query, "limit": 3})
        except Exception as e:
            logger.error(f"Failed to invoke fetch_reviews_webscraping_ai: {e}")
            reviews = ["Could not fetch reviews."]
            
        reviews_map[name] = reviews
        
        # Enrich recommendation with reviews
        place["reviews_data"] = reviews
        updated_recommendations.append(place)
        
    return {
        "reviews": reviews_map,
        "recommendations": updated_recommendations
    }


# ==========================================
# STEP 7: ANALYZE REVIEWS NODE
# ==========================================

async def analyze_reviews_node(state: AgentState) -> dict:
    """
    STEP 7: Analyze Reviews Node
    
    Uses LLM to analyze the fetched reviews for the top candidates and pick the winner.
    """
    logger.info("🧠 STEP 7: Analyze Reviews Node")
    
    recommendations = state.get("recommendations", [])
    user_query = state.get("user_query", "")
    
    if not recommendations:
        return {"is_complete": True}
        
    # Prepare data for LLM
    candidates_data = []
    for place in recommendations:
        candidates_data.append({
            "name": place.get("name"),
            "rating": place.get("rating"),
            "reviews_count": place.get("reviews_count"),
            "reviews": place.get("reviews_data", [])
        })
        
    llm = get_llm(temperature=0.2)
    
    system_prompt = """You are an expert local guide. Your task is to pick the SINGLE BEST place from the provided candidates based on the user's request and the actual content of the reviews.

Analyze the reviews for:
1. Sentiment (are people happy?)
2. Specific mentions relevant to the user's query (e.g., "clean", "friendly", "equipment")
3. Red flags (e.g., "hidden fees", "rude staff")

Output Format:
Return the name of the winning place exactly as it appears in the input.
Then, provide a short "Why this is the winner" explanation.
"""

    user_prompt = f"""
User Query: "{user_query}"

Candidates and their Reviews:
{json.dumps(candidates_data, indent=2)}

Who is the winner?
"""

    # We use a structured output or just parse the text. Let's use text for flexibility and parse the name.
    # Actually, let's use structured output for the name to be safe.
    
    class WinnerSelection(BaseModel):
        winner_name: str = Field(description="The exact name of the winning place")
        explanation: str = Field(description="Why this place was chosen based on reviews")
        key_pros: list[str] = Field(description="List of 3 key positive points from reviews")
        key_cons: list[str] = Field(description="List of 1-2 negative points or warnings (if any)")

    structured_llm = llm.with_structured_output(WinnerSelection)
    
    try:
        decision = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        winner_name = decision.winner_name
        explanation = decision.explanation
        
        # Find the winner object
        winner = None
        for place in recommendations:
            if place.get("name") == winner_name:
                winner = place
                break
        
        if not winner:
            # Fallback fuzzy match
            for place in recommendations:
                if place.get("name") in winner_name or winner_name in place.get("name"):
                    winner = place
                    break
        
        if not winner:
            winner = recommendations[0] # Fallback to first
            
        # Format the final output
        result_text = f"""
🏆 **WINNER SELECTED:** {winner.get('name')}

⭐ **Rating:** {winner.get('rating')}/5.0 ({winner.get('reviews_count')} reviews)
📍 **Address:** {winner.get('address')}

**Why it won:**
{explanation}

**Key Highlights:**
{chr(10).join([f"- {pro}" for pro in decision.key_pros])}

**Things to Note:**
{chr(10).join([f"- {con}" for con in decision.key_cons])}
"""
        
        return {
            "current_step": "analyze_reviews",
            "is_complete": True,
            "recommendations": [winner], # Return just the winner
            "messages": state["messages"] + [AIMessage(content=result_text)]
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_reviews_node: {e}")
        # Fallback
        return {
            "current_step": "analyze_reviews",
            "is_complete": True,
            "recommendations": [recommendations[0]],
            "messages": state["messages"] + [AIMessage(content="Could not analyze reviews in detail. Returning top rated option.")]
        }
