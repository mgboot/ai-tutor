import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

async def main():
    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY_4o",
        "AZURE_OPENAI_ENDPOINT_4o",
        "AZURE_OPENAI_DEPLOYMENT_4o",
        "AZURE_OPENAI_API_KEY_o1",
        "AZURE_OPENAI_ENDPOINT_o1",
        "AZURE_OPENAI_DEPLOYMENT_o1"
    ]
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} environment variable not found. Please add it to your .env file.")
            return

    # Create an instance of the kernel
    kernel = Kernel()

    # Register the primary GPT-4o model for conversation
    primary_model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_4o")
    primary_service_id = "gpt4o"

    kernel.add_service(
        AzureChatCompletion(
            service_id=primary_service_id,
            deployment_name=primary_model_id,
            api_key=os.getenv("AZURE_OPENAI_API_KEY_4o"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o")
        )
    )

    # Register the secondary o1 model for deep reasoning
    secondary_model_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_o1")
    secondary_service_id = "o1-model"

    kernel.add_service(
        AzureChatCompletion(
            service_id=secondary_service_id,
            deployment_name=secondary_model_id,
            api_key=os.getenv("AZURE_OPENAI_API_KEY_o1"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_o1")
        )
    )

    # Create a deep reasoning function using the o1 model
    deep_reasoning_prompt = """
    You are an advanced reasoning system specializing in breaking down complex problems inductively and helping to identify likely sources of misunderstanding.
    Think step-by-step to analyze the following problem:

    Problem: {{$problem}}

    1. Break down the problem into key components
    2. Identify the core concepts and relationships
    3. Apply relevant reasoning frameworks
    4. Provide a detailed, well-structured analysis
    5. Conclude with a clear statement of what the potential misunderstandings might be
    6. Suggest a specific keyword or two on which the student's misunderstanding may be hinging
    """

    deep_reasoning_config = PromptTemplateConfig(
        template=deep_reasoning_prompt,
        name="deep_reasoning",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="problem", description="The complex problem to analyze", is_required=True),
        ],
        execution_settings={
            secondary_service_id: {
                "model": secondary_model_id,
                "temperature": 0.1,
                "max_tokens": 3000
            }
        }
    )

    # Add the deep reasoning function to the kernel
    deep_reasoning_function = kernel.add_function(
        function_name="DeepReasoning",
        plugin_name="ReasoningPlugin",
        prompt_template_config=deep_reasoning_config,
        description="Use this function for complex reasoning tasks that require breaking down the user's intent or solving multi-step problems."
    )

    # Define system message for the tutor
    system_message = """
    You are an intelligent tutor named AI Tutor. Your primary goal is to help users understand complex topics 
    and provide clear, educational responses to their questions, especially in evaluating quiz answers.

    When faced with the challenge of evaluating a quiz answer or student response that appears to be incorrect, use the DeepReasoning
    function to identify what material or concepts the student may have misunderstood in order to arrive at the result they got.

    Be sure to suggest topics for the student to review, and/or concepts the student might have missed.
    This is more important than simply correcting the student's answer.
    
    Guidelines:
    - Be friendly, patient, and educational in your responses
    - Use the DeepReasoning function when you need to reason through what a student may have misunderstood
    - Explain concepts clearly using examples when helpful
    - When using DeepReasoning, integrate its insights into your response naturally
    - Always maintain a helpful, tutoring tone
    """

    # Create a chat function for the primary model
    chat_prompt = """{{$chat_history}}\nUser: {{$user_input}}\nTutor:"""

    chat_config = PromptTemplateConfig(
        template=chat_prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="chat_history", description="The conversation history", is_required=True),
            InputVariable(name="user_input", description="The user's input", is_required=True),
        ],
        execution_settings={
            primary_service_id: {
                "model": primary_model_id,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    )

    chat_function = kernel.add_function(
        function_name="Chat",
        plugin_name="TutorPlugin",
        prompt_template_config=chat_config,
    )

    # Initialize chat history
    chat_history = ChatHistory()
    chat_history.add_system_message(system_message)

    # Configure function calling settings
    function_execution_settings = OpenAIChatPromptExecutionSettings(
        service_id=primary_service_id,
        ai_model_id=primary_model_id,
        temperature=0.7,
        max_tokens=1000,
        function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True)
    )

    # Prepare kernel arguments with settings
    arguments = KernelArguments(settings=function_execution_settings)
    arguments["chat_history"] = chat_history

    async def chat_with_tutor(user_input: str) -> str:
        """Chat with the tutor and get a response."""
        # Update the arguments with the user input
        arguments["user_input"] = user_input
        
        # Invoke the chat function
        result = await kernel.invoke(chat_function, arguments=arguments)
        
        # Update the chat history
        chat_history.add_user_message(user_input)
        chat_history.add_assistant_message(str(result))
        
        return str(result)
    
    # Console-based chat interface
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
        response = await chat_with_tutor(user_input)
        print(f"\nAI Tutor: {response}")

if __name__ == "__main__":
    asyncio.run(main())
