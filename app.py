from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import os
import json
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()

# Setup Azure OpenAI client
client = openai.AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_4o"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_4o"),
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_4o")

# Initialize FastAPI 
app = FastAPI(
    title="Azure OpenAI Streaming Chat",
    version="1.0",
    description="Simple streaming chat API with Azure OpenAI"
)

# Define request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Endpoint for streaming chat responses"""
    
    # Convert Pydantic models to dictionaries for OpenAI
    messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
    
    async def generate():
        try:
            # Create streaming response from Azure OpenAI
            stream = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                stream=True,
                temperature=0.7,
            )
            
            # Stream each chunk back to the client
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Send completion signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    """Endpoint for non-streaming chat responses"""
    
    # Convert Pydantic models to dictionaries for OpenAI
    messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
    
    try:
        # Get response from Azure OpenAI
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7,
        )
        
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)