import os
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional

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

# Define agent names
TUTOR_NAME = "Tutor"
REASONING_NAME = "Reasoning"

class TutorAgentManager:
    """
    A class that manages the AI Tutor agents and provides methods
    for streaming conversation with them.
    """
    
    def __init__(self, use_env_vars: bool = True):
        """
        Initialize the TutorAgentManager.
        
        Args:
            use_env_vars: Whether to load configuration from environment variables
        """
        self.kernel = None
        self.chat = None
        self.tutor_agent = None
        self.reasoning_agent = None
        
        if use_env_vars:
            self._setup_from_env()
    
    def _setup_from_env(self):
        """Set up the agents using environment variables."""
        # Create kernel and add models
        self.kernel, _, _, _, _ = self._setup_kernel_with_models()
        
        # Create agents
        self.tutor_agent = self._create_tutor_agent()
        self.reasoning_agent = self._create_reasoning_agent()
        
        # Create the agent group chat
        self._setup_agent_chat()
    
    def _setup_kernel_with_models(self):
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
        
        self.kernel = kernel
        return kernel, primary_service_id, primary_model_id, secondary_service_id, secondary_model_id
    
    def _create_tutor_agent(self):
        """Create a tutor agent that can interact with students."""
        return ChatCompletionAgent(
            kernel=self.kernel,
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
    
    def _create_reasoning_agent(self):
        """Create a reasoning agent that can analyze problems in depth."""
        return ChatCompletionAgent(
            kernel=self.kernel,
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
    
    def _setup_agent_chat(self):
        """Set up the agent chat with selection and termination strategies."""
        # Define selection function - simplified
        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=f"""
Choose the next agent to respond based on the conversation context.
Only respond with the name of the agent without explanation.

Choose from:
- {TUTOR_NAME}
- {REASONING_NAME}

Rules:
- If the message is from a user, the {TUTOR_NAME} should respond
- If the message is from the {TUTOR_NAME} and explicitly asks for reasoning help, the {REASONING_NAME} should respond
- If the message is from the {REASONING_NAME}, the {TUTOR_NAME} should respond
- Never choose the same agent that just spoke unless there's a clear handoff

HISTORY:
{{{{$lastmessage}}}}
""",
        )

        # Define termination function - simplified
        termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=f"""
Determine if this agent conversation should end and return to the user.
Answer with "yes" or "no" only.

Respond with "yes" if:
1. The {TUTOR_NAME} has provided a complete response to the user's question
2. The {TUTOR_NAME} has responded after getting input from the {REASONING_NAME}
3. The conversation between agents has reached a natural conclusion

Respond with "no" only if:
- The {TUTOR_NAME} has explicitly asked the {REASONING_NAME} for help and the {REASONING_NAME} hasn't yet responded

HISTORY:
{{{{$lastmessage}}}}
""",
        )

        history_reducer = ChatHistoryTruncationReducer(target_count=5)

        # Create the agent group chat with simpler configuration
        self.chat = AgentGroupChat(
            agents=[self.tutor_agent, self.reasoning_agent],
            selection_strategy=KernelFunctionSelectionStrategy(
                initial_agent=self.tutor_agent,
                function=selection_function,
                kernel=self.kernel,
                result_parser=lambda result: str(result.value[0]).strip() if result.value and result.value[0] else TUTOR_NAME,
                history_variable_name="lastmessage",
                history_reducer=history_reducer,
            ),
            termination_strategy=KernelFunctionTerminationStrategy(
                agents=[self.tutor_agent],
                function=termination_function,
                kernel=self.kernel,
                result_parser=lambda result: "yes" in str(result.value[0]).lower() if result.value and result.value[0] else True,
                history_variable_name="lastmessage",
                maximum_iterations=3,
                history_reducer=history_reducer,
            ),
        )
    
    async def reset(self):
        """Reset the chat history."""
        if self.chat:
            await self.chat.reset()
    
    async def add_message(self, message: str):
        """Add a message to the chat."""
        if self.chat:
            # Ensure chat is ready for a new message
            self.chat.is_complete = False
            await self.chat.add_chat_message(message=message)
    
    async def stream_response(self) -> AsyncGenerator[Dict[str, str], None]:
        """
        Stream responses from the agents.
        
        Yields:
            Dict containing 'agent' and 'content' for each chunk
        """
        if not self.chat:
            yield {"error": "Chat not initialized"}
            return

        last_agent = None
        
        try:
            async for response in self.chat.invoke_stream():
                if response is None or not response.name:
                    continue
                
                # If this is a new agent speaking, indicate that
                if last_agent != response.name:
                    last_agent = response.name
                    # Only yield the agent name for the first chunk
                    yield {"agent": response.name, "content": ""}
                
                # Yield the content chunk
                if response.content:
                    yield {"agent": response.name, "content": response.content}
                    
        except Exception as e:
            if "Chat is already complete" in str(e):
                # Expected when conversation turn ends normally
                pass
            else:
                yield {"error": str(e)}
        
        # Reset the completion state for the next conversation turn
        self.chat.is_complete = False

# Create a singleton instance for use across the application
tutor_manager = TutorAgentManager()

async def process_chat_message(message: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process a chat message through the tutor system and stream the response.
    
    Args:
        message: The user's message
        
    Yields:
        Dictionary with agent and content information for each chunk
    """
    # Add the message to the chat
    await tutor_manager.add_message(message)
    
    # Stream the responses
    async for chunk in tutor_manager.stream_response():
        yield chunk

async def reset_chat():
    """Reset the chat history."""
    await tutor_manager.reset()
