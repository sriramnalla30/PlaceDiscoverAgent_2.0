from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from typing import Optional
from app.models import (
    SearchRequest,
    SearchResult,
    AgentActivationRequest,
    AgentChatRequest,
    HumanApprovalRequest,
    AgentResponse,
    PlaceType,
    InitialQueryRequest
)
from app.agent.graph import negotiator_agent, get_thread_config, visualize_graph
from app.agent.tools import search_places
from app.config import settings
import uuid
import json
import logging

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ==========================================
# INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Negotiator Agent API",
    description="AI Agent for place discovery and negotiation with HITL support",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "service": "Negotiator Agent API",
        "version": "1.0.0",
        "environment": settings.environment,
        "workflow": visualize_graph()
    }


# ==========================================
# ENDPOINT 1: INITIAL SEARCH (SERP API)
# ==========================================

@app.post("/search", response_model=list[SearchResult])
async def search_endpoint(request: SearchRequest):
    """
    Initial place search using SERP API.
    Returns raw results before agent mode.
    
    Flow: User searches → Gets results → Can activate agent mode
    """
    try:
        logger.info(f"Search request: {request.city}, {request.place_type}")
        
        # Use SERP tool
        results = search_places.invoke({
            "city": request.city,
            "place_type": request.place_type.value,
            "query": request.query
        })
        
        if not results:
            raise HTTPException(status_code=404, detail="No places found")
        
        # Convert to SearchResult models
        search_results = [
            SearchResult(
                name=place.get("name", "Unknown"),
                address=place.get("address", ""),
                phone=place.get("phone"),
                rating=place.get("rating"),
                price_level=place.get("price_level"),
                type=place.get("type", request.place_type.value)
            )
            for place in results
            if "error" not in place
        ]
        
        logger.info(f"Found {len(search_results)} places")
        return search_results
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 2: START AGENT (YOUR DESIGN: STEP 1-2-3)
# ==========================================

@app.post("/agent/start")
async def start_agent(request: InitialQueryRequest):
    """
    STEP 1-3 Implementation from MY_DESIGN_README
    
    STEP 1: User submits natural language query
    STEP 2: Responder LLM analyzes intent
    STEP 3: Shows "Searching SERPSTACK API with params"
    """
    try:
        from langchain_groq import ChatGroq
        from pydantic import BaseModel, Field
        
        # STEP 1: Receive user query
        user_query = request.user_query
        logger.info(f"STEP 1: User query received: {user_query}")
        
        # STEP 2: Responder LLM - Parse query into structured format
        class ParsedQuery(BaseModel):
            """Structured query extraction"""
            city: str = Field(description="City name extracted from query")
            area: str = Field(description="Area/locality within city, if mentioned")
            place_type: str = Field(description="Type of place (gym, cafe, restaurant, etc.)")
            intent: str = Field(description="What user wants: best reviewed, negotiation, pricing info, etc.")
            budget: Optional[str] = Field(None, description="Budget mentioned, if any")
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=settings.groq_api_key  # GROQ_API_KEY
        )
        
        structured_llm = llm.with_structured_output(ParsedQuery)
        
        responder_prompt = f"""
        You are a query parser for a place discovery system.
        
        User query: "{user_query}"
        
        Extract:
        - city: The city name
        - area: Specific area/locality if mentioned (otherwise use city name)
        - place_type: Type of place (gym, cafe, restaurant, hotel, etc.)
        - intent: What the user wants (best reviewed, negotiation, pricing, etc.)
        - budget: Any budget mentioned (extract number if present)
        
        Examples:
        "I need gyms in koramangala in bangalore" → city: bangalore, area: koramangala, type: gym
        "Find cafes in MG Road" → city: MG Road, area: MG Road, type: cafe
        """
        
        logger.info("STEP 2: Responder LLM analyzing query...")
        parsed_query = structured_llm.invoke(responder_prompt)
        
        logger.info(f"STEP 2: Parsed - City: {parsed_query.city}, Area: {parsed_query.area}, Type: {parsed_query.place_type}")
        
        # Generate thread_id
        thread_id = str(uuid.uuid4()) #here we are saving to memory using thread_id
        
        # STEP 3: Prepare for SERPSTACK API call
        # Combine city + area for better search results
        search_location = f"{parsed_query.area} {parsed_query.city}" if parsed_query.area and parsed_query.area != parsed_query.city else parsed_query.city
        
        # Construct a clean search query
        final_query = f"{parsed_query.place_type} in {search_location}"
        
        search_params = {
            "city": search_location,
            "place_type": parsed_query.place_type,
            "query": final_query
        }
        
        logger.info(f"STEP 3: Searching SERPSTACK API with: Query='{final_query}'")
        
        # Call SERPSTACK (STEP 4 will execute in streaming)
        results = search_places.invoke(search_params)
        
        if not results or any("error" in str(r) for r in results):
            logger.warning("SERPSTACK returned no results or error")
            results = []
        
        # Initialize agent state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "city": parsed_query.city,
            "place_type": parsed_query.place_type,
            "user_intent": parsed_query.intent,
            "budget": None,
            "serp_results": results,
            "thread_id": thread_id,
            "iteration": 0,
            "is_complete": False,
            "show_all": False,  # Will be set by Revisor node
            "user_query": user_query,  # Store original query
            "parsed_params": {
                "city": parsed_query.city,
                "area": parsed_query.area,
                "type": parsed_query.place_type
            }
        }
        
        config = get_thread_config(thread_id)
        
        # Save initial state to memory (do NOT execute agent yet)
        # Frontend will immediately call /stream which will start execution
        logger.info(f"Saving initial state for thread {thread_id}")
        await negotiator_agent.aupdate_state(config, initial_state)
        
        return {
            "thread_id": thread_id,
            "message": "Agent initialized successfully",
            "step": "search_params_ready",
            "parsed_params": {
                "city": parsed_query.city,
                "area": parsed_query.area,
                "type": parsed_query.place_type,
                "intent": parsed_query.intent
            },
            "places_found": len(results)
        }
    
    except Exception as e:
        logger.error(f"Start agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 3: ACTIVATE AGENT MODE
# ==========================================

@app.post("/agent/activate", response_model=AgentResponse)
async def activate_agent(request: AgentActivationRequest):
    """
    Activates agent mode with user's search results.
    Starts the LangGraph workflow and returns thread_id.
    
    Will pause at HITL checkpoint for human approval.
    """
    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        logger.info(f"Activating agent for thread: {thread_id}")
        
        # Convert SearchResult models to dicts
        serp_results = [result.model_dump() for result in request.search_results]
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=request.user_intent)],
            "city": request.search_results[0].address.split(",")[-1].strip() if request.search_results else "Unknown",
            "place_type": request.search_results[0].type if request.search_results else "unknown",
            "user_intent": request.user_intent,
            "budget": request.budget,
            "serp_results": serp_results,
            "tavily_reviews": [],
            "human_approved": False,
            "human_notes": None,
            "route": "",
            "current_step": "start",
            "initial_analysis": [],
            "shop_responses": [],
            "refined_analysis": [],
            "iteration": 0,
            "recommendations": None,
            "is_complete": False,
            "thread_id": thread_id
        }
        
        # Run agent until HITL interrupt
        config = get_thread_config(thread_id)
        
        # Invoke will run until interrupt_before=["human_review"]
        state = await negotiator_agent.ainvoke(initial_state, config)
        
        # Get last message from agent
        last_message = state["messages"][-1].content if state["messages"] else "Agent initialized"
        
        return AgentResponse(
            message=last_message,
            thread_id=thread_id,
            data={
                "route": state.get("route"),
                "current_step": state.get("current_step"),
                "serp_results": serp_results
            },
            requires_approval=True,  # HITL checkpoint
            is_complete=False
        )
    
    except Exception as e:
        logger.error(f"Agent activation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 3: HUMAN APPROVAL (HITL)
# ==========================================

@app.post("/agent/approve", response_model=AgentResponse)
async def approve_and_continue(request: HumanApprovalRequest):
    """
    Human-in-the-Loop approval endpoint.
    User reviews data and approves/modifies before agent continues.
    
    Resumes graph execution after interrupt.
    """
    try:
        logger.info(f"HITL approval for thread: {request.thread_id}")
        
        config = get_thread_config(request.thread_id)
        
        # Get current state
        current_state = negotiator_agent.get_state(config)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Prepare resume input
        resume_input = None
        
        if request.approved:
            # User approved, continue with existing data
            resume_input = {
                "human_approved": True,
                "human_notes": request.user_notes
            }
            
            # If user modified results, update them
            if request.modified_results:
                resume_input["serp_results"] = [
                    result.model_dump() for result in request.modified_results
                ]
        else:
            # User rejected, return error
            return AgentResponse(
                message="Analysis cancelled by user",
                thread_id=request.thread_id,
                is_complete=True
            )
        
        # Resume execution
        state = await negotiator_agent.ainvoke(resume_input, config)
        
        # Get recommendations if complete
        is_complete = state.get("is_complete", False)
        last_message = state["messages"][-1].content if state["messages"] else "Processing..."
        
        return AgentResponse(
            message=last_message,
            thread_id=request.thread_id,
            data={
                "recommendations": state.get("recommendations"),
                "current_step": state.get("current_step")
            },
            requires_approval=False,
            is_complete=is_complete
        )
    
    except Exception as e:
        logger.error(f"Approval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 4: CONVERSATIONAL CHAT
# ==========================================

@app.post("/agent/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Continue conversation with agent after recommendations.
    User can ask follow-up questions, request refinements.
    
    Uses existing thread_id to maintain context (memory).
    """
    try:
        logger.info(f"Chat request for thread: {request.thread_id}")
        
        config = get_thread_config(request.thread_id)
        
        # Get current state
        current_state = negotiator_agent.get_state(config)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found. Please start a new session.")
        
        # Add user message to existing conversation
        new_input = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Invoke agent with new message
        state = await negotiator_agent.ainvoke(new_input, config)
        
        # Get response
        last_message = state["messages"][-1].content if state["messages"] else "No response"
        
        return AgentResponse(
            message=last_message,
            thread_id=request.thread_id,
            data={
                "recommendations": state.get("recommendations"),
                "current_step": state.get("current_step")
            },
            is_complete=state.get("is_complete", False)
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 5: STREAMING RESPONSE
# ==========================================

@app.get("/agent/stream")
async def stream_agent(thread_id: str):
    """
    Streaming endpoint for real-time agent responses with verbose logging.
    Returns Server-Sent Events (SSE) for frontend with detailed API/LLM calls.
    Uses existing thread_id from /agent/start.
    """
    
    async def event_generator():
        try:
            config = get_thread_config(thread_id)
            
            # Get current state
            state_snapshot = await negotiator_agent.aget_state(config)
            
            if not state_snapshot or not state_snapshot.values:
                error_data = json.dumps({"type": "error", "error": "Thread not found"})
                yield f"data: {error_data}\n\n"
                return
            
            logger.info(f"Streaming for thread: {thread_id}")
            
            # Send initial state info
            init_data = json.dumps({
                "type": "init",
                "city": state_snapshot.values.get("city"),
                "place_type": state_snapshot.values.get("place_type"),
                "serp_results_count": len(state_snapshot.values.get("serp_results", []))
            })
            yield f"data: {init_data}\n\n"
            
            # Show the SerpStack results first (from STEP 4)
            serp_results = state_snapshot.values.get("serp_results", [])
            if serp_results:
                results_preview = json.dumps({
                    "type": "data_fetched",
                    "source": "SerpStack",
                    "count": len(serp_results),
                    "sample": serp_results[:3] if len(serp_results) > 3 else serp_results
                })
                yield f"data: {results_preview}\n\n"
                logger.info(f"Showed {len(serp_results)} SerpStack results")
            
            # Check if we need to resume or if it's complete
            if not state_snapshot.next:
                # Graph hasn't started yet or is complete
                if state_snapshot.values.get("is_complete"):
                    complete_data = json.dumps({
                        "type": "complete",
                        "recommendations": state_snapshot.values.get("recommendations", []),
                        "all_places": state_snapshot.values.get("serp_results", []),
                        "iteration": state_snapshot.values.get("iteration", 0)
                    })
                    yield f"data: {complete_data}\n\n"
                    return
                
                # Graph hasn't started - kick it off now
                logger.info("Starting agent execution...")
            
            # Stream state updates using stream_mode="updates" to track node changes
            previous_step = state_snapshot.values.get("current_step", "")
            
            async for chunk in negotiator_agent.astream(None, config, stream_mode="updates"):
                # chunk is a dict like: {"node_name": updated_state}
                for node_name, node_state in chunk.items():
                    logger.info(f"Node update: {node_name}")
                    
                    # Node started
                    start_data = json.dumps({
                        "type": "node_start",
                        "node": node_name,
                        "timestamp": str(uuid.uuid4())[:8]
                    })
                    yield f"data: {start_data}\n\n"
                    
                    # Check for specific node types to extract details
                    current_step = node_state.get("current_step", "")
                    
                    # Router node - show decision
                    if node_name == "revisor":
                        router_decision = node_state.get("show_all", False)
                        decision_data = json.dumps({
                            "type": "router_decision",
                            "requires_negotiation": False, # Revisor doesn't decide negotiation yet in this path
                            "reason": "Showing ALL results" if router_decision else "Showing BEST result"
                        })
                        yield f"data: {decision_data}\n\n"
                    
                    # Review Extraction node
                    elif node_name == "review_extraction":
                        reviews = node_state.get("reviews", {})
                        review_data = json.dumps({
                            "type": "reviews_fetched",
                            "count": len(reviews),
                            "reviews": reviews  # Send the full reviews map {PlaceName: [Review1, Review2]}
                        })
                        yield f"data: {review_data}\n\n"
                    
                    # Analyze Reviews node
                    elif node_name == "analyze_reviews":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content"):
                                llm_data = json.dumps({
                                    "type": "llm_response",
                                    "model": "llama-3.3-70b",
                                    "content": str(last_msg.content)
                                })
                                yield f"data: {llm_data}\n\n"

                    # Analyze node - show LLM analysis
                    elif node_name == "analyze_node":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content"):
                                llm_data = json.dumps({
                                    "type": "llm_response",
                                    "model": "gemini-2.5-flash",
                                    "content": str(last_msg.content)[:500]  # First 500 chars
                                })
                                yield f"data: {llm_data}\n\n"
                    
                    # Negotiate node - show simulation
                    elif node_name == "negotiate_node":
                        negotiations = node_state.get("negotiations", [])
                        if negotiations:
                            nego_data = json.dumps({
                                "type": "negotiation",
                                "count": len(negotiations),
                                "sample": negotiations[-1] if negotiations else None
                            })
                            yield f"data: {nego_data}\n\n"
                    
                    # Reflexion node - show iteration
                    elif node_name == "reflexion_node":
                        iteration = node_state.get("iteration", 0)
                        reflexion_data = json.dumps({
                            "type": "reflexion",
                            "iteration": iteration,
                            "status": "Refining recommendations..."
                        })
                        yield f"data: {reflexion_data}\n\n"
                    
                    # Node completed
                    end_data = json.dumps({
                        "type": "node_end",
                        "node": node_name,
                        "iteration": node_state.get("iteration", 0)
                    })
                    yield f"data: {end_data}\n\n"
                
                # Check for interrupts after each update
                current_state = await negotiator_agent.aget_state(config)
                
                # Check if interrupted for HITL
                if current_state.next and current_state.next == ("human_review_node",):
                    interrupt_data = json.dumps({
                        "type": "interrupt",
                        "node": "human_review_node",
                        "serp_results": current_state.values.get("serp_results", []),
                        "places_count": len(current_state.values.get("serp_results", []))
                    })
                    yield f"data: {interrupt_data}\n\n"
                    logger.info("HITL interrupt detected")
                    return
                
                # Check if complete
                if not current_state.next and current_state.values.get("is_complete"):
                    complete_data = json.dumps({
                        "type": "complete",
                        "recommendations": current_state.values.get("recommendations", []),
                        "all_places": current_state.values.get("serp_results", []),
                        "iteration": current_state.values.get("iteration", 0)
                    })
                    yield f"data: {complete_data}\n\n"
                    logger.info("Agent completed")
                    return
        
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_data = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ==========================================
# ENDPOINT 6: GET CONVERSATION HISTORY
# ==========================================

@app.get("/agent/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieve conversation history for a thread.
    Shows all checkpoints and state transitions.
    """
    try:
        config = get_thread_config(thread_id)
        
        # Get state history
        history = []
        for state in negotiator_agent.get_state_history(config):
            history.append({
                "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                "step": state.metadata.get("step", 0),
                "current_node": state.values.get("current_step"),
                "is_complete": state.values.get("is_complete", False),
                "message_count": len(state.values.get("messages", []))
            })
        
        return {
            "thread_id": thread_id,
            "total_checkpoints": len(history),
            "history": history
        }
    
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 7: RESET CONVERSATION
# ==========================================

@app.delete("/agent/reset/{thread_id}")
async def reset_thread(thread_id: str):
    """
    Clear conversation history for a thread.
    Useful for starting fresh.
    """
    try:
        # Note: LangGraph doesn't have direct delete,
        # but starting with new thread_id effectively resets
        return {
            "message": f"Thread {thread_id} marked for reset. Start new conversation with different thread_id.",
            "new_thread_id": str(uuid.uuid4())
        }
    
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 9: QUICK START (SEARCH + ACTIVATE)
# ==========================================

@app.post("/agent/start", response_model=AgentResponse)
async def quick_start_agent(
    city: str,
    place_type: str,
    budget: Optional[float] = None,
    user_preferences: Optional[str] = None
):
    """
    Convenience endpoint that combines /search and /agent/activate.
    Searches for places and immediately activates agent mode.
    
    Usage:
        POST /agent/start?city=Bangalore&place_type=gym&budget=5000&user_preferences=Looking for good equipment
    """
    try:
        logger.info(f"Quick start: {city}, {place_type}")
        
        # Step 1: Search for places
        search_req = SearchRequest(
            city=city,
            place_type=PlaceType(place_type.lower()),
            query=user_preferences
        )
        
        results = search_places.invoke({
            "city": search_req.city,
            "place_type": search_req.place_type.value,
            "query": search_req.query
        })
        
        if not results or any("error" in r for r in results):
            raise HTTPException(
                status_code=404, 
                detail=f"No places found or API error: {results[0].get('error', 'Unknown') if results else 'No results'}"
            )
        
        # Convert to SearchResult models
        search_results = [
            SearchResult(
                name=place.get("name", "Unknown"),
                address=place.get("address", ""),
                phone=place.get("phone"),
                rating=place.get("rating"),
                price_level=place.get("price_level"),
                type=place.get("type", place_type)
            )
            for place in results
            if "error" not in place
        ]
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No valid places found")
        
        logger.info(f"Found {len(search_results)} places")
        
        # Step 2: Activate agent with results
        user_intent = user_preferences or f"Find the best {place_type} in {city}"
        if budget:
            user_intent += f" within budget of ₹{budget}"
        
        activation_req = AgentActivationRequest(
            search_results=search_results,
            user_intent=user_intent,
            budget=budget
        )
        
        # Activate agent
        return await activate_agent(activation_req)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
