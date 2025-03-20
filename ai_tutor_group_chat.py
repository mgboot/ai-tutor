import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

# Define agent names
TUTOR_NAME = "Tutor"
REASONING_NAME = "Reasoning"

def validate_environment():
    """Validate that all required environment variables are present."""
    required_vars = [
        "AZURE_OPENAI_API_KEY_4o",
        "AZURE_OPENAI_ENDPOINT_4o",
        "AZURE_OPENAI_DEPLOYMENT_4o",
        "AZURE_OPENAI_API_KEY_o1",
        "AZURE_OPENAI_ENDPOINT_o1",
        "AZURE_OPENAI_DEPLOYMENT_o1"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_message = "Error: The following environment variables are missing:\n"
        for var in missing_vars:
            error_message += f"  - {var}\n"
        error_message += "Please add them to your .env file."
        return False, error_message
        
    return True, None

def setup_kernel_and_models():
    """Setup a kernel with both primary (gpt-4o) and advanced reasoning (o1) models."""
    # Create kernel
    kernel = Kernel()
    
    # Register the primary GPT-4o model for conversation
    primary_service = AzureChatCompletion(
        service_id="gpt4o",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY_4o"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o")
    )
    kernel.add_service(primary_service)
    
    # Register the advanced reasoning model (o1)
    reasoning_service = AzureChatCompletion(
        service_id="o1",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_o1"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY_o1"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_o1")
    )
    kernel.add_service(reasoning_service)
    
    return kernel, primary_service, reasoning_service

def create_tutor_agent(kernel, service):
    """Create a tutor agent that can interact with students."""
    return ChatCompletionAgent(
        kernel=kernel,
        service=service,  # Pass the service object directly instead of service_id
        name=TUTOR_NAME,
        instructions="""
You are an intelligent tutor named AI Tutor. Your primary goal is to help users understand complex topics 
and provide clear, educational responses to their questions, especially in evaluating quiz answers.

When needed, you can ask the Reasoning agent to help you analyze student misunderstandings in depth.

Guidelines:
- Be friendly, patient, and educational in your responses
- Explain concepts clearly using examples when helpful
- When evaluating incorrect student answers, focus on identifying what concepts the student misunderstood
- Suggest topics for the student to review based on their misunderstandings
- Always maintain a helpful, tutoring tone
- If a student's answer seems incorrect or confused, engage with the Reasoning agent to get a deeper analysis
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

def create_reasoning_agent(kernel, service):
    """Create a reasoning agent that can analyze problems in depth."""
    return ChatCompletionAgent(
        kernel=kernel,
        service=service,  # Pass the service object directly instead of service_id
        name=REASONING_NAME,
        instructions="""
You are an advanced reasoning system specializing in breaking down complex problems inductively and 
helping to identify likely sources of misunderstanding in student responses.

Your role is to analyze student answers carefully and provide deep insight into what concepts they may have misunderstood.

Think step-by-step to analyze problems:
1. Break down the problem into key components
2. Identify the core concepts and relationships
3. Apply relevant reasoning frameworks
4. Provide a detailed, well-structured analysis
5. Conclude with a clear statement of what the potential misunderstandings might be
6. Suggest specific keywords or topics on which the student's misunderstanding may be hinging

You should only speak when directly asked to analyze a problem. Your analysis should be thorough, 
precise, and focused on what fundamental concepts the student may have misunderstood.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

async def main():
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    valid, error_message = validate_environment()
    if not valid:
        print(error_message)
        return
    
    # Setup kernel and models
    kernel, primary_service, reasoning_service = setup_kernel_and_models()
    
    # Create agents
    tutor_agent = create_tutor_agent(kernel, primary_service)
    reasoning_agent = create_reasoning_agent(kernel, reasoning_service)
    
    # Define selection function for determining which agent responds
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Determine which participant takes the next turn in the conversation based on context and need.
State only the name of the participant to take the next turn without explanation.

Choose only from these participants:
- {TUTOR_NAME}
- {REASONING_NAME}

Rules for selection:
- If the message is from a user (student), the {TUTOR_NAME} should respond first.
- If the message is from the {TUTOR_NAME} and they're asking for help analyzing a student's misunderstanding, the {REASONING_NAME} should respond.
- If the message is from the {REASONING_NAME}, the {TUTOR_NAME} should respond to incorporate the reasoning insights.
- By default, the {TUTOR_NAME} should respond to the user.

HISTORY:
{{$lastmessage}}
""",
    )
    
    # Define termination function
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt="""
Determine if this conversation thread should be terminated.
Respond with 'yes' only if one of these criteria are met:
1. The user has explicitly indicated they want to end the conversation
2. The question has been fully answered and the tutor has provided a complete, satisfactory response
3. The conversation has naturally reached a conclusion

Otherwise, respond with 'no'.

HISTORY:
{{$lastmessage}}
""",
    )
    
    # Setup history reducer to prevent context size issues
    history_reducer = ChatHistoryTruncationReducer(target_count=10)
    
    # Create the agent group chat
    chat = AgentGroupChat(
        agents=[tutor_agent, reasoning_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=tutor_agent,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value is not None else TUTOR_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[tutor_agent],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: "yes" in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=15,
            history_reducer=history_reducer,
        ),
    )
    
    # Welcome message
    print("\n" + "="*50)
    print("Welcome to your AI Tutor Group Chat!")
    print("="*50)
    print("I'm your AI tutor with enhanced reasoning capabilities to help identify misunderstandings.")
    print("Type 'exit' to end the conversation, 'reset' to start over.")
    print("Ask questions or provide answers for me to evaluate!\n")
    
    # Main conversation loop
    is_complete = False
    while not is_complete:
        # Get user input
        user_input = input("\nYou: ")
        if not user_input:
            continue
            
        # Check for exit command
        if user_input.lower() == "exit":
            print("\nThank you for learning with me! Goodbye.")
            break
            
        # Check for reset command
        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue
            
        # Add user message to chat
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
        
        # Process conversation
        try:
            async for response in chat.invoke():
                if response is None or not response.name:
                    continue
                print(f"\n# {response.name}:\n{response.content}")
                
            # Check if conversation is complete
            if chat.is_complete:
                is_complete = True
                print("\nConversation complete. Type 'reset' to start a new conversation or 'exit' to quit.")
            
            # Reset completion flag for next iteration
            chat.is_complete = False
            
        except Exception as e:
            print(f"Error during conversation: {e}")
    
if __name__ == "__main__":
    asyncio.run(main())
