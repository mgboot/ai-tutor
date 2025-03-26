from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any

# Import our custom tutor pattern
from tutor_pattern import process_chat_message, reset_chat

# Load environment variables
load_dotenv()

# Initialize FastAPI 
app = FastAPI(
    title="AI Tutor API",
    version="1.0",
    description="Streaming AI Tutor API with FastAPI"
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Tutor API is running"}

# Add a route to handle WebSocket connection attempts
@app.get("/ws/{path:path}")
async def handle_websocket(path: str):
    """Handle WebSocket connection attempts with a proper response"""
    return Response(
        content=json.dumps({"error": "WebSocket connections not supported"}),
        media_type="application/json",
        status_code=404
    )

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Endpoint for streaming chat responses from AI Tutor"""
    
    # Extract the last user message from the conversation history
    last_user_message = None
    for msg in reversed(chat_request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'No user message found'})}\n\n"]),
            media_type="text/event-stream"
        )
    
    async def generate():
        try:
            # Process the message through our tutor system
            last_agent = None
            
            async for chunk in process_chat_message(last_user_message):
                if "error" in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    continue
                
                # If this is a new agent, send the agent name
                if last_agent != chunk["agent"]:
                    last_agent = chunk["agent"]
                    yield f"data: {json.dumps({'agent': chunk['agent']})}\n\n"
                
                # Send the content chunk
                if chunk["content"]:
                    yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            
            # Send completion signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/chat/reset")
async def reset():
    """Reset the chat history"""
    await reset_chat()
    return {"status": "success", "message": "Chat reset successfully"}

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    """Endpoint for non-streaming chat responses - not recommended for tutor agent"""
    return {"error": "Please use the streaming endpoint /chat/stream for better experience with the AI Tutor"}

if __name__ == "__main__":
    # Configure logging to reduce noise from 404s
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    
    print("Starting FastAPI server on port 8000...")
    print("Make sure your Streamlit client connects to http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000, log_config=log_config)