import os
import json
import requests
from typing import Dict, Any, List

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API endpoint - Updated to use port 8001
API_URL = "http://localhost:8001"  # Changed from 8000 to 8001

# For debugging
def debug_request(url, payload, error=None):
    """Log request information for debugging"""
    print(f"Request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    if error:
        print(f"Error: {error}")

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session when a user connects."""
    # Verify server is running
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code != 200:
            await cl.Message(
                content=f"⚠️ Warning: API server returned status {response.status_code}. The backend may not be running correctly. Make sure FastAPI is running on port 8001.",
                author="System"
            ).send()
    except requests.exceptions.ConnectionError:
        await cl.Message(
            content=f"⚠️ Error: Cannot connect to API server at {API_URL}. Please ensure the backend is running with `python app.py` on port 8001.",
            author="System"
        ).send()
    
    # Send a welcome message
    await cl.Message(
        content="""
# Welcome to AI Tutor! 👋

I'm your AI tutor with enhanced reasoning capabilities to help identify misunderstandings.

**How I can help you:**
- Answer questions about complex topics
- Evaluate your answers to quiz questions
- Provide detailed explanations with examples
- Identify misconceptions in your understanding

Ask a question or provide an answer for me to evaluate!
        """,
        author="AI Tutor",
    ).send()
    
    # Initialize empty message history
    cl.user_session.set("messages", [])

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    # Get current message history
    messages = cl.user_session.get("messages", [])
    
    # Add user message to history
    messages.append({"role": "user", "content": message.content})
    cl.user_session.set("messages", messages)
    
    # Prepare to send to backend API
    payload = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages]
    }
    
    # Create message elements to update
    msg = cl.Message(author="AI Tutor", content="Thinking...")
    await msg.send()
    
    # Stream the response from the API
    current_agent = None
    full_response = ""
    
    try:
        # For debugging
        debug_url = f"{API_URL}/chat/stream"
        
        with requests.post(
            debug_url,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"}
        ) as response:
            if response.status_code == 404:
                await msg.update(content=f"Error: API endpoint not found. Make sure the backend server is running and has the correct endpoints defined.")
                debug_request(debug_url, payload, "404 Not Found")
                return
                
            if response.status_code != 200:
                await msg.update(content=f"Error: Received status code {response.status_code} from API. Response: {response.text}")
                debug_request(debug_url, payload, f"Status {response.status_code}: {response.text}")
                return
                
            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode("utf-8")
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Handle agent identification
                        if "agent" in data:
                            new_agent = data["agent"]
                            if current_agent != new_agent:
                                # If this is a new agent and not the first one, add a separator
                                if current_agent and new_agent != "Tutor":
                                    full_response += f"\n\n**{new_agent}'s Analysis:**\n"
                                # Set the author for the first time or when changing agents
                                if not current_agent:
                                    await msg.update(author=new_agent)
                                current_agent = new_agent
                        
                        # Handle content chunks
                        if "content" in data:
                            full_response += data["content"]
                            await msg.update(content=full_response)
                        
                        # Handle errors
                        if "error" in data:
                            await cl.Message(content=f"Error: {data['error']}", author="System").send()
                            return
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}, Data: {data_str}")
                        continue
                    
    except requests.exceptions.ConnectionError:
        await msg.update(content="Error: Cannot connect to the API server. Please ensure the backend is running.")
        return
    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        return
    
    # If we didn't get any response, show error
    if not full_response:
        await msg.update(content="Error: No response received from the server.")
        return
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": full_response})
    cl.user_session.set("messages", messages)
    
    # Add a button to reset the conversation
    await cl.Button(
        name="reset",
        label="Reset Conversation"
    ).send()

@cl.action_callback("reset")
async def reset_conversation(action):
    """Reset the conversation."""
    # Clear local message history
    cl.user_session.set("messages", [])
    
    # Call backend reset endpoint
    try:
        response = requests.post(f"{API_URL}/chat/reset")
        if response.status_code == 200:
            await cl.Message(content="Conversation has been reset. Let's start fresh!", author="System").send()
        else:
            await cl.Message(content=f"Failed to reset conversation: {response.text}", author="System").send()
    except Exception as e:
        await cl.Message(content=f"Error resetting conversation: {str(e)}", author="System").send()

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates."""
    print(f"Settings updated: {settings}")
