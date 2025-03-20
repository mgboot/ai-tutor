from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from tutor.config import TutorConfig

def create_kernel_functions(kernel: Kernel, config: TutorConfig) -> None:
    """Create and register kernel functions."""
    
    # Create deep reasoning function using secondary model
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
            config.secondary_service_id: {
                "model": config.secondary_deployment,
                "temperature": 0.1,
                "max_tokens": 3000
            }
        }
    )

    # Add deep reasoning function to kernel
    kernel.add_function(
        function_name="DeepReasoning",
        plugin_name="ReasoningPlugin",
        prompt_template_config=deep_reasoning_config,
        description="Use this function for complex reasoning tasks that require breaking down the user's intent or solving multi-step problems."
    )

    # Create chat function using primary model
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
            config.primary_service_id: {
                "model": config.primary_deployment,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    )

    # Add chat function to kernel
    kernel.add_function(
        function_name="Chat",
        plugin_name="TutorPlugin",
        prompt_template_config=chat_config,
    )
