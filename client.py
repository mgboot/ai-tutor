import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="AI Tutor Chat", page_icon="🧠")

# Initialize session state to store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message_stream(messages):
    """
    Send messages to the AI Tutor API and stream the response.
    
    Args:
        messages (list): List of message dictionaries with role and content.
    """
    # Display a placeholder for the streaming response
    message_placeholder = st.empty()
    full_response = ""
    current_agent = None
    
    # Format the request payload
    payload = {
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    }
    
    # Make streaming request
    try:
        with requests.post("http://localhost:8000/chat/stream", 
                         json=payload, 
                         stream=True,
                         headers={'Accept': 'text/event-stream'}) as response:
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Handle new agent identification
                            if 'agent' in data:
                                current_agent = data['agent']
                                if full_response:
                                    # If we're switching agents, add a divider
                                    full_response += f"\n\n**{current_agent}**:\n"
                                else:
                                    full_response += f"**{current_agent}**:\n"
                                    
                            # Handle content chunks
                            if 'content' in data:
                                full_response += data['content']
                                # Update the placeholder with the accumulated response
                                message_placeholder.markdown(full_response)
                                
                            # Handle errors
                            if 'error' in data:
                                st.error(f"Error: {data['error']}")
                                
                        except json.JSONDecodeError:
                            continue
        
        # Return the full response once streaming is complete
        return full_response
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Error connecting to server."

def reset_chat():
    """Reset the chat by clearing session state and calling reset endpoint."""
    try:
        response = requests.post("http://localhost:8000/chat/reset")
        if response.status_code == 200:
            # Clear the session state
            st.session_state.messages = []
            st.success("Chat reset successfully!")
        else:
            st.error("Failed to reset chat on server.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    """Main function for the Streamlit app"""
    st.title("AI Tutor Chat")
    st.subheader("Your intelligent learning assistant")
    
    # Add a reset button
    if st.button("Reset Chat"):
        reset_chat()
        st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = send_message_stream(st.session_state.messages)
            if response:
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()