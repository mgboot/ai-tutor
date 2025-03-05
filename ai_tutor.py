import asyncio
import os
import dotenv
import logging

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from logging_utils import log_message, log_flow, log_from_agent, log_separator
from local_python_plugin import LocalPythonPlugin

# Load environment variables
dotenv.load_dotenv()

# Configuration
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Agent names
QUIZ_CREATOR_NAME = "QuizCreator"
EVALUATOR_NAME = "Evaluator"
INFO_REVIEWER_NAME = "InfoReviewer"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Subject to tutor on
DEFAULT_SUBJECT = "Electronics and Circuit Design"
# Current topic being studied
DEFAULT_TOPIC = "Basic circuit components including resistors, capacitors, and transistors"

def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_key=azure_openai_api_key,
            api_version=azure_openai_api_version,
        )
    )
    
    # Add the local Python plugin for data analysis
    kernel.add_plugin(plugin_name="LocalCodeExecutionTool", plugin=LocalPythonPlugin())
    
    return kernel

async def main():
    # Get subject and topic from user or use defaults
    subject = input(f"Enter the subject to tutor on (default: {DEFAULT_SUBJECT}): ") or DEFAULT_SUBJECT
    topic = input(f"Enter the current topic being studied (default: {DEFAULT_TOPIC}): ") or DEFAULT_TOPIC
    
    # Create the Quiz Creator agent
    agent_quiz_creator = ChatCompletionAgent(
        service_id=QUIZ_CREATOR_NAME,
        kernel=_create_kernel_with_chat_completion(QUIZ_CREATOR_NAME),
        name=QUIZ_CREATOR_NAME,
        instructions=f"""
            You are an expert {QUIZ_CREATOR_NAME} for {subject}. 
            
            Your responsibility is to:
            1. Create assessment questions to evaluate student understanding
            2. Focus questions on the current topic: {topic}
            3. Create questions that test different levels of understanding
            4. When creating follow-up questions, focus on areas where the student showed weakness
            
            Guidelines for assessments:
            - Keep questions clear and concise
            - Include a mix of conceptual and practical questions
            - Provide 3-5 questions in each assessment
            - For initial assessment, cover the breadth of the topic
            - For follow-up assessment, focus specifically on weak areas identified by the Evaluator
            
            You are working with other specialized agents: {EVALUATOR_NAME} and {INFO_REVIEWER_NAME}.
            
            First assessment: Create a comprehensive initial assessment on {topic}.
            Follow-up assessment: After the {EVALUATOR_NAME} identifies weaknesses and the {INFO_REVIEWER_NAME} provides materials,
            create a focused assessment on those specific weak areas.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=QUIZ_CREATOR_NAME,
            temperature=0.4,
            max_tokens=1500,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    # Create the Evaluator agent
    agent_evaluator = ChatCompletionAgent(
        service_id=EVALUATOR_NAME,
        kernel=_create_kernel_with_chat_completion(EVALUATOR_NAME),
        name=EVALUATOR_NAME,
        instructions=f"""
            You are an expert {EVALUATOR_NAME} for {subject}.
            
            Your responsibility is to:
            1. Carefully analyze student responses to assessment questions
            2. Identify specific knowledge gaps or misunderstandings
            3. Determine which concepts the student struggles with most
            4. Provide a clear, specific diagnosis of where the student needs additional help
            
            Guidelines for evaluation:
            - Look for patterns in student mistakes
            - Identify fundamental concepts that might be misunderstood
            - Be specific about which sub-topics need reinforcement
            - Rank the severity of knowledge gaps to prioritize remediation
            - Use examples from the student's answers to support your assessment
            
            You are working with other specialized agents: {QUIZ_CREATOR_NAME} and {INFO_REVIEWER_NAME}.
            
            When analyzing student responses, use this structure:
            1. Overall assessment of student understanding
            2. Specific knowledge gaps identified
            3. Primary concept requiring remediation
            4. Evidence from student responses
            
            If you notice a pattern like always getting questions about a specific component wrong (e.g., capacitors),
            be sure to highlight this as the primary area needing remediation.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=EVALUATOR_NAME,
            temperature=0.2,
            max_tokens=1500,
            # Fix: Use the correct API for SK 1.18.2
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )
    
    # Create the Info Reviewer agent
    agent_info_reviewer = ChatCompletionAgent(
        service_id=INFO_REVIEWER_NAME,
        kernel=_create_kernel_with_chat_completion(INFO_REVIEWER_NAME),
        name=INFO_REVIEWER_NAME,
        instructions=f"""
            You are an expert {INFO_REVIEWER_NAME} for {subject}.
            
            Your responsibility is to:
            1. Provide targeted educational materials based on the Evaluator's assessment
            2. Create concise, clear explanations for concepts the student struggles with
            3. Present information in multiple ways to support different learning styles
            4. Focus on practical examples that reinforce theoretical concepts
            
            Guidelines for providing materials:
            - Keep explanations clear, focused, and at appropriate depth
            - Use analogies to help with difficult concepts
            - Include diagrams and visualizations (described textually)
            - Present both theoretical foundation and practical applications
            - Provide 1-2 examples that demonstrate the concept in action
            
            You are working with other specialized agents: {QUIZ_CREATOR_NAME} and {EVALUATOR_NAME}.
            
            Structure your educational materials as:
            1. Introduction to the concept
            2. Core theoretical explanation
            3. Visual representation (described)
            4. Practical example
            5. Common misconceptions clarified
            
            When the {EVALUATOR_NAME} identifies a specific weak area (e.g., capacitors),
            focus your materials entirely on that topic rather than providing broad coverage.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id=INFO_REVIEWER_NAME,
            temperature=0.5,
            max_tokens=2000,
            # Fix: Use the correct API for SK 1.18.2
            function_choice_behavior=FunctionChoiceBehavior.Required(),
        ),
    )

    # Define the selection strategy for turn-taking
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
        Determine which participant takes the next turn in a conversation based on the current state and flow.
        State only the name of the participant to take the next turn.
        
        Choose only from these participants:
        - {QUIZ_CREATOR_NAME}
        - {EVALUATOR_NAME}
        - {INFO_REVIEWER_NAME}

        Follow these rules when selecting the next participant:
        - After initial user input describing a tutoring need, {QUIZ_CREATOR_NAME} creates the first assessment
        - After {QUIZ_CREATOR_NAME} provides an assessment, wait for student responses (user input)
        - After student responds to questions, {EVALUATOR_NAME} analyzes their responses
        - After {EVALUATOR_NAME} identifies gaps, {INFO_REVIEWER_NAME} provides educational materials
        - After {INFO_REVIEWER_NAME} provides materials, {QUIZ_CREATOR_NAME} creates a follow-up assessment
        - The cycle continues: assessment → student response → evaluation → materials → follow-up assessment

        History:
        {{$history}}
        """,
    )

    # Define termination condition
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
            Examine the conversation history and determine if the tutoring session should end.
            Respond with "yes" ONLY if one of these criteria is met:
            1. The student has explicitly asked to end the session
            2. The student has demonstrated mastery of all topics after at least one full cycle of assessment-evaluation-review-reassessment
            3. The conversation has exceeded 3 full cycles of tutoring
            
            Otherwise, respond with "no" to continue the session.

            History:
            {{$history}}
            """,
    )

    # Create the agent group chat
    chat = AgentGroupChat(
        agents=[agent_quiz_creator, agent_evaluator, agent_info_reviewer],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=_create_kernel_with_chat_completion("selection"),
            result_parser=lambda result: str(result.value[0]) if result.value is not None else QUIZ_CREATOR_NAME,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_quiz_creator, agent_evaluator, agent_info_reviewer],
            function=termination_function,
            kernel=_create_kernel_with_chat_completion("termination"),
            result_parser=lambda result: "yes" in str(result.value[0]).lower(),
            history_variable_name="history",
            maximum_iterations=15,
        ),
    )

    # Initial prompt to start the tutoring session
    initial_prompt = f"""
    I need help learning about {topic} as part of my {subject} studies. 
    Please create an assessment to check my understanding, then help me with any concepts I'm struggling with.
    """
    
    print("\n===== AI TUTOR SESSION STARTED =====")
    print(f"Subject: {subject}")
    print(f"Topic: {topic}")
    print("\nInitiating first assessment...\n")
    
    log_separator()
    log_message("Starting tutoring session")
    log_flow("System", "")
    print(f"{initial_prompt}\n")

    await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=initial_prompt))

    is_complete = False
    while not is_complete:
        # Get next agent response
        async for response in chat.invoke():
            log_separator()
            log_message(f"Invoking {response.name} agent")
            log_from_agent(response.name)
            print(f"{response.content}\n")

        if chat.is_complete:
            is_complete = True
            print("\n===== AI TUTOR SESSION COMPLETED =====")
            break
            
        # Get student response
        user_input = input("\nYour response: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            print("\n===== AI TUTOR SESSION ENDED BY USER =====")
            break
            
        if user_input.lower() == "reset":
            await chat.reset()
            print("[Tutoring session has been reset]")
            continue
            
        log_separator()
        log_flow("Student", "")
        print(f"{user_input}\n")
        
        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

if __name__ == "__main__":
    asyncio.run(main())
