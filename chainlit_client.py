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
    
    # Check if this is a reset command
    if message.content.lower() in ["reset", "restart", "clear"]:
        await reset_chat()
        return
    
    # Prepare to send to backend API
    payload = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages]
    }
    
    # Create message elements to update - Updated for newer Chainlit API
    msg = cl.Message(content="Thinking...", author="AI Tutor")
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
                # Using new approach for updating messages
                new_msg = cl.Message(content=f"Error: API endpoint not found. Make sure the backend server is running and has the correct endpoints defined.", author="System")
                await new_msg.send()
                await msg.remove()  # Remove the thinking message
                debug_request(debug_url, payload, "404 Not Found")
                return
                
            if response.status_code != 200:
                # Using new approach for updating messages
                new_msg = cl.Message(content=f"Error: Received status code {response.status_code} from API. Response: {response.text}", author="System")
                await new_msg.send()
                await msg.remove()  # Remove the thinking message
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
                                # Set the author - handle author change differently
                                if not current_agent:
                                    current_agent = new_agent
                                    # Create a new message with the correct author
                                    await msg.remove()  # Remove the thinking message
                                    msg = cl.Message(content="", author=new_agent)
                                    await msg.send()
                        
                        # Handle content chunks
                        if "content" in data:
                            full_response += data["content"]
                            # Update content - using the newer Chainlit API approach
                            msg.content = full_response
                            await msg.update()
                        
                        # Handle errors
                        if "error" in data:
                            await cl.Message(content=f"Error: {data['error']}", author="System").send()
                            return
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}, Data: {data_str}")
                        continue
                    
    except requests.exceptions.ConnectionError:
        # Create a new message instead of updating
        await msg.remove()
        await cl.Message(content="Error: Cannot connect to the API server. Please ensure the backend is running.", author="System").send()
        return
    except Exception as e:
        # Create a new message instead of updating
        await msg.remove()
        await cl.Message(content=f"Error: {str(e)}", author="System").send()
        return
    
    # If we didn't get any response, show error
    if not full_response:
        await msg.remove()
        await cl.Message(content="Error: No response received from the server.", author="System").send()
        return
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": full_response})
    cl.user_session.set("messages", messages)
    
    # # Add a message with reset instructions instead of a button
    # await cl.Message(
    #     content="*Type 'reset' to start a new conversation*",
    #     author="System"
    # ).send()

# Simple function to reset the chat
async def reset_chat():
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
