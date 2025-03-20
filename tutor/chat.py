from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from tutor.config import TutorConfig

class TutorChat:
    """Class to handle chat interactions with the tutor."""
    
    def __init__(self, kernel: Kernel, config: TutorConfig = None):
        """Initialize the tutor chat with a kernel and optional config."""
        self.kernel = kernel
        self.config = config
        
        # Initialize chat history with system message
        self.chat_history = ChatHistory()
        system_message = config.system_message if config else self._get_default_system_message()
        self.chat_history.add_system_message(system_message)
        
        # Get service IDs
        self.primary_service_id = config.primary_service_id if config else "gpt4o"
        self.primary_model_id = config.primary_deployment if config else None
        
    def _get_default_system_message(self) -> str:
        """Return the default system message if no config is provided."""
        return """
        You are an intelligent tutor named AI Tutor. Your primary goal is to help users understand complex topics 
        and provide clear, educational responses to their questions.
        
        Guidelines:
        - Be friendly, patient, and educational in your responses
        - Explain concepts clearly using examples when helpful
        - Always maintain a helpful, tutoring tone
        """
    
    async def get_response(self, user_input: str) -> str:
        """Get a response from the tutor for the given user input."""
        # Configure function calling settings
        function_execution_settings = OpenAIChatPromptExecutionSettings(
            service_id=self.primary_service_id,
            ai_model_id=self.primary_model_id,
            temperature=0.7,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True)
        )
        
        # Prepare kernel arguments
        arguments = KernelArguments(settings=function_execution_settings)
        arguments["chat_history"] = self.chat_history
        arguments["user_input"] = user_input
        
        # Invoke the chat function
        chat_function = self.kernel.plugins["TutorPlugin"]["Chat"] 
        result = await self.kernel.invoke(chat_function, arguments=arguments)
        
        # Update chat history
        self.chat_history.add_user_message(user_input)
        self.chat_history.add_assistant_message(str(result))
        
        return str(result)
