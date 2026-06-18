import sys
import json
import logging
from pathlib import Path
import asyncio

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import your LangGraph pipeline
from src.pipeline.graph import build_graph
from src.pipeline.state import create_initial_state

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LangGraph agent once at startup
try:
    app_graph = build_graph()
    logger.info("CRAG pipeline graph initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CRAG pipeline: {e}")
    app_graph = None

# Initialize FastAPI
app = FastAPI(
    title="CRAG AI Agent API",
    description="Production API for LangGraph Corrective RAG System",
    version="1.0.0"
)

# Enable CORS for frontend integration
# NOTE: allow_credentials=True is incompatible with allow_origins=["*"]
# In production, replace "*" with your actual frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, example="How does Corrective RAG handle irrelevant documents?")
    thread_id: str = Field(default="default-thread", example="user-123-session-abc")
    force_web_search: bool = Field(default=False)

# --- Streaming Generator ---
async def generate_stream(request: ChatRequest):
    """
    Asynchronous generator that yields Server-Sent Events (SSE).
    We use LangGraph's astream() to yield node updates and tokens.
    """
    # Create the initial state dictionary
    initial_state = create_initial_state(request.query)
    
    # Configuration for session state tracking
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # We use stream_mode="updates" to track which node is executing
        async for chunk in app_graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                
                # We strip out complex objects (like LangChain Documents) to avoid JSON serialization errors
                safe_state = {}
                if "grade" in node_state:
                    safe_state["grade"] = node_state["grade"]
                if "source" in node_state:
                    safe_state["source"] = node_state["source"]
                if "generation" in node_state:
                    safe_state["generation"] = node_state["generation"]
                if "hallucination" in node_state:
                    safe_state["hallucination"] = node_state["hallucination"]

                # Construct the event payload
                event_data = {
                    "event": "node_update",
                    "node": node_name,
                    "state_updates": safe_state
                }
                
                # Format strictly for SSE: data: <json>\n\n
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Small sleep to yield control back to the event loop
                await asyncio.sleep(0.01)

        # Send a final completion event
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    except asyncio.CancelledError:
        logger.warning(f"Client disconnected. Halting generation for thread {request.thread_id}.")
        # Gracefully handle the disconnect to save LLM tokens
        raise
    except Exception as e:
        logger.error(f"Error during graph execution: {e}", exc_info=True)
        error_data = {"event": "error", "message": "An internal error occurred. Please try again."}
        yield f"data: {json.dumps(error_data)}\n\n"

# --- Endpoints ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint that returns Server-Sent Events.
    """
    if app_graph is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check server logs.")
    # The X-Accel-Buffering header prevents Nginx from buffering the stream in production
    return StreamingResponse(
        generate_stream(request), 
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Load Balancers."""
    return {"status": "healthy", "service": "CRAG Agent"}

if __name__ == "__main__":
    import uvicorn
    # Run the ASGI server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)