import os
import asyncio
from dotenv import load_dotenv

from tutor.config import load_configuration
from tutor.models import create_kernel_with_models
from tutor.functions import create_kernel_functions
from tutor.chat import TutorChat

async def main():
    # Load environment variables and configuration
    load_dotenv()
    config = load_configuration()
    
    if not config.is_valid():
        print("Error: Missing required environment variables. Please check your .env file.")
        return
        
    # Create kernel with all required AI models
    kernel = create_kernel_with_models(config)
    
    # Create and register kernel functions
    create_kernel_functions(kernel, config)
    
    # Initialize the tutor chat interface
    tutor_chat = TutorChat(kernel)
    
    # Start the console-based chat interface
    print("\n" + "="*50)
    print("Welcome to your AI Tutor chat!")
    print("="*50)
    print("I'm your AI tutor with enhanced inductive reasoning capabilities.\nWhat would you like to discuss today?")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for chatting! Goodbye.")
            break
            
        print("\nAI Tutor is thinking...")
        response = await tutor_chat.get_response(user_input)
        print(f"\nAI Tutor: {response}")

if __name__ == "__main__":
    asyncio.run(main())
