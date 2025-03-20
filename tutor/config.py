import os
from dataclasses import dataclass

@dataclass
class TutorConfig:
    # Primary model (GPT-4o) configuration
    primary_api_key: str
    primary_endpoint: str
    primary_deployment: str
    
    # Secondary model (o1) configuration
    secondary_api_key: str
    secondary_endpoint: str
    secondary_deployment: str
    
    # Parameters with default values must come after those without defaults
    primary_service_id: str = "gpt4o"
    secondary_service_id: str = "o1-model"
    
    # System message for the tutor
    system_message: str = """
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
    
    def is_valid(self) -> bool:
        """Check if all required configuration values are present."""
        return all([
            self.primary_api_key,
            self.primary_endpoint,
            self.primary_deployment,
            self.secondary_api_key,
            self.secondary_endpoint,
            self.secondary_deployment
        ])

def load_configuration() -> TutorConfig:
    """Load configuration from environment variables."""
    return TutorConfig(
        # Primary model (GPT-4o)
        primary_api_key=os.getenv("AZURE_OPENAI_API_KEY_4o", ""),
        primary_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o", ""),
        primary_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_4o", ""),
        
        # Secondary model (o1)
        secondary_api_key=os.getenv("AZURE_OPENAI_API_KEY_o1", ""),
        secondary_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_o1", ""),
        secondary_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_o1", "")
    )
