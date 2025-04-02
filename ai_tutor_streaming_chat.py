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
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

"""
This sample demonstrates a streaming version of the AI Tutor Group Chat.
It creates a chat interface where you can interact with a Tutor and a Reasoning 
agent that work together to provide educational assistance.
"""

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

def setup_kernel_with_models():
    """Create and configure a kernel with both primary and secondary models."""
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
    
    return kernel

async def main():
    # Load environment variables
    load_dotenv()

    # Validate environment
    valid, error_message = validate_environment()
    if not valid:
        print(error_message)
        return
    
    kernel = setup_kernel_with_models()
    
    # Create Tutor agent
    tutor_agent = ChatCompletionAgent(
        kernel=kernel,
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

    # Create Reasoning agent
    reasoning_agent = ChatCompletionAgent(
        kernel=kernel,
        name=REASONING_NAME,
        instructions="""
You are an advanced reasoning system specializing in breaking down complex problems inductively and helping to identify likely sources of misunderstanding.
Think step-by-step to analyze problems:
    1. Break down the problem into key components
    2. Work backwards from the incorrect answer to identify where student departed from correct method
    3. Show how, with a minimal number of steps, one would arrive at that *incorrect* answer
    4. Conclude with a clear statement of what the potential misunderstanding might be
    5. Suggest a specific keyword or two on which the student's misunderstanding may be hinging
    """,
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Define selection function - simplified
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {TUTOR_NAME}
- {REASONING_NAME}

Rules:
- If RESPONSE is from user input, it is {TUTOR_NAME}'s turn.
- If RESPONSE is by {TUTOR_NAME} and explicitly asks for reasoning help, the {REASONING_NAME}'s turn.
- If RESPONSE is by {REASONING_NAME}, it is {TUTOR_NAME}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define a termination function where the reviewer signals completion with "yes".
    termination_keyword = "yes"

    # Define termination function - simplified
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )
# Determine if this agent conversation should end and return to the user.
# Answer with "yes" or "no" only.

# Respond with "yes" if:
# 1. The {TUTOR_NAME} has provided a complete response to the user's question
# 2. The {TUTOR_NAME} has responded after getting input from the {REASONING_NAME}
# 3. The conversation between agents has reached a natural conclusion

# Respond with "no" only if:
# - The {TUTOR_NAME} has explicitly asked the {REASONING_NAME} for help and the {REASONING_NAME} hasn't yet responded

    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    # Create the agent group chat with simpler configuration
    chat = AgentGroupChat(
        agents=[tutor_agent, reasoning_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=tutor_agent,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value and result.value[0] else TUTOR_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[tutor_agent],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=3,
            history_reducer=history_reducer,
        ),
    )

    print("\n" + "="*50)
    print("Welcome to your AI Tutor Chat with Streaming!")
    print("="*50)
    print("I'm your AI tutor with enhanced reasoning capabilities to help identify misunderstandings.")
    print("Type 'exit' to end the conversation, 'reset' to start over.")
    print("Ask questions or provide answers for me to evaluate!\n")

    is_complete = False
    while not is_complete:
        print()
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nThank you for learning with me! Goodbye.")
            is_complete = True
            break

        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue

        # # Create a new chat instance for each interaction to avoid state issues
        # if hasattr(chat, '_chat_history') and chat._chat_history:
        #     chat.is_complete = False

        # Add user message to chat
        await chat.add_chat_message(message=user_input)

        # Process streaming responses
        try:
            last_agent = None
            print()
            
            # Stream responses from agents
            async for response in chat.invoke_stream():
                if response is None or not response.name:
                    continue
                
                # If this is a new agent speaking, print the agent name header
                if last_agent != response.name:
                    if last_agent is not None:
                        print()  # Add line break between agents
                    print(f"\n# {response.name}:", end="", flush=True)
                    last_agent = response.name
                    print()  # Line break after agent name
                
                # Print the content chunk
                if response.content:
                    print(response.content, end="", flush=True)
                    
        except Exception as e:
            if "Chat is already complete" in str(e):
                # This is normal and expected when conversation turns end
                pass
            else:
                print(f"\nError during chat: {e}")

        # Reset the completion state for the next conversation turn
        chat.is_complete = False

if __name__ == "__main__":
    asyncio.run(main())
