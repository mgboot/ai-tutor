import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")

# Initialize session state to store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message_stream(messages):
    """
    Send messages to the chat API and stream the response.
    
    Args:
        messages (list): List of message dictionaries with role and content.
    """
    # Display a placeholder for the streaming response
    message_placeholder = st.empty()
    full_response = ""
    
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
                            if 'content' in data:
                                full_response += data['content']
                                # Update the placeholder with the accumulated response
                                message_placeholder.markdown(full_response)
                        except json.JSONDecodeError:
                            continue
        
        # Return the full response once streaming is complete
        return full_response
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Error connecting to server."

def main():
    """Main function for the Streamlit app"""
    st.title("Azure OpenAI Chat")
    st.subheader("Chat with GPT-4o")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
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