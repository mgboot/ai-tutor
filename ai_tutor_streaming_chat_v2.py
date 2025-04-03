import asyncio
import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any
from enum import Enum

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
from semantic_kernel.functions.kernel_arguments import KernelArguments

"""
This sample demonstrates a streaming version of the AI Tutor Group Chat.
It creates a chat interface where you can interact with a Tutor and a Reasoning 
agent that work together to provide educational assistance with pattern detection.
"""

# Define agent names
TUTOR_NAME = "Tutor"
EVALUATOR_NAME = "Evaluator"
QUIZ_CREATOR_NAME = "QuizCreator"

# Define termination keyword
termination_keyword = "complete"

# Define states for the tutor system
class TutorState(Enum):
    CHAT = "chat"
    QUIZ = "quiz"
    REVIEW = "review"

class StudentTracker:
    """Tracks student performance and identifies patterns in learning."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.questions = []
        self.answers = []
        self.correct_answers = []
        self.is_correct = []
        self.topics = []
        self.consecutive_wrong = 0
        self.topic_performance = {}
        
    def add_question(self, question: str, correct_answer: str, topics: List[str]):
        """Add a new question to the tracker."""
        self.questions.append(question)
        self.correct_answers.append(correct_answer)
        self.topics.append(topics)
        
    def add_answer(self, answer: str, is_correct: bool):
        """Add a student's answer to the tracker."""
        self.answers.append(answer)
        self.is_correct.append(is_correct)
        
        # Update consecutive wrong counter
        if is_correct:
            self.consecutive_wrong = 0
        else:
            self.consecutive_wrong += 1
            
            # Update topic performance
            if len(self.topics) > 0 and len(self.is_correct) <= len(self.topics):
                question_topics = self.topics[len(self.is_correct) - 1]
                for topic in question_topics:
                    if topic not in self.topic_performance:
                        self.topic_performance[topic] = {"correct": 0, "wrong": 0}
                    self.topic_performance[topic]["wrong"] += 1
        
        # If correct, update topic performance
        if is_correct and len(self.topics) > 0 and len(self.is_correct) <= len(self.topics):
            question_topics = self.topics[len(self.is_correct) - 1]
            for topic in question_topics:
                if topic not in self.topic_performance:
                    self.topic_performance[topic] = {"correct": 0, "wrong": 0}
                self.topic_performance[topic]["correct"] += 1
    
    def get_problem_topics(self) -> List[str]:
        """Return topics where the student is struggling."""
        problem_topics = []
        for topic, stats in self.topic_performance.items():
            total = stats["correct"] + stats["wrong"]
            if total >= 3 and stats["wrong"] / total >= 0.5:
                problem_topics.append(topic)
        return problem_topics
    
    def should_trigger_review(self) -> bool:
        """Check if student needs a review based on performance."""
        return self.consecutive_wrong >= 3 or len(self.get_problem_topics()) > 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the student's performance."""
        total_questions = len(self.is_correct)
        total_correct = sum(1 for x in self.is_correct if x)
        
        return {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "accuracy": total_correct / total_questions if total_questions > 0 else 0,
            "consecutive_wrong": self.consecutive_wrong,
            "topic_performance": self.topic_performance,
            "problem_topics": self.get_problem_topics()
        }

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
    
    # Create student tracker
    student_tracker = StudentTracker()
    
    # Track current state
    current_state = TutorState.CHAT
    
    # Create Tutor agent
    tutor_agent = ChatCompletionAgent(
        kernel=kernel,
        name=TUTOR_NAME,
        instructions="""
You are an intelligent tutor named AI Tutor. Your primary goal is to help users understand complex topics 
and provide clear, educational responses to their questions, especially in evaluating quiz answers.

You have multiple roles:
1. QUIZ CREATOR - You can create multiple choice quizzes on topics the student is studying
2. EVALUATOR - You assess student answers and track their understanding
3. INFO REVIEWER - When patterns of misunderstanding are identified, you provide targeted materials

When needed, you can ask the {{Ev}} agent to analyze student misunderstandings in depth.

Guidelines:
- Be friendly, patient, and educational in your responses
- Present ONLY ONE question at a time in quiz mode - wait for the user to answer before providing the next question
- Each question should be labeled with a topic or category
- Each question should have four options (A, B, C, D)
- When evaluating answers, clearly state if the answer is right or wrong
- If a student gets multiple answers wrong in a row (3+), engage the {EVALUATOR_NAME} agent
- After {EVALUATOR_NAME} identifies knowledge gaps, provide targeted materials focused on those gaps
- Create follow-up questions specifically addressing the areas of weakness
- Always maintain a helpful, tutoring tone
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Create Reasoning agent
    reasoning_agent = ChatCompletionAgent(
        kernel=kernel,
        name=EVALUATOR_NAME,
        instructions="""
You are an advanced evaluator reasoning system specializing in analyzing student performance patterns to identify knowledge gaps.

When analyzing quiz performance, follow these steps:
1. Review the student's quiz history and pattern of answers
2. Identify specific topics or concepts where the student consistently makes mistakes
3. Analyze the nature of the mistakes to determine likely misconceptions
4. Consider what fundamental concept the student might be misunderstanding
5. Recommend specific topics for review that would address these knowledge gaps
6. Suggest an approach for presenting this information that would help clarify the misconception

For example, if a student consistently misidentifies terrier breeds as sporting dogs, identify:
- What specific characteristics they might be confused about
- What distinguishing features they're overlooking
- What conceptual framework would help them better categorize the breeds

Your goal is to provide actionable insights, not just identify errors.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Define selection function with enhanced logic
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE, CHAT HISTORY, and STUDENT PERFORMANCE to choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {TUTOR_NAME}
- {EVALUATOR_NAME}

Rules:
- If RESPONSE is from user input, it is {TUTOR_NAME}'s turn.
- If STUDENT PERFORMANCE shows 3+ consecutive wrong answers, choose {EVALUATOR_NAME} to analyze.
- If RESPONSE is by {EVALUATOR_NAME}, it is {TUTOR_NAME}'s turn.
- If a pattern of topic-specific errors is detected, choose {EVALUATOR_NAME} for analysis.

CURRENT STATE: {{{{$current_state}}}}
CONSECUTIVE WRONG ANSWERS: {{{{$consecutive_wrong}}}}
PROBLEM TOPICS: {{{{$problem_topics}}}}

CHAT HISTORY:
{{{{$chat_history}}}}

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define termination function - simplified
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.
If the {EVALUATOR_NAME} agent has just provided analysis and the Tutor hasn't responded yet, respond with: no.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=10)

    # Create the agent group chat with enhanced configuration
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
            maximum_iterations=5,
            history_reducer=history_reducer,
        ),
    )

    print("\n" + "="*50)
    print("Welcome to your AI Tutor Chat with Pattern Detection!")
    print("="*50)
    print("I'm your AI tutor with enhanced reasoning capabilities to help identify misunderstandings.")
    print("\nWhat topic would you like to learn about today? I can create quizzes and provide information on many subjects.")
    print("\nSome example topics you might choose:")
    print("- Dog breeds (classification, characteristics, history)")
    print("- Geometry (shapes, theorems, calculations)")
    print("- World history (civilizations, events, figures)")
    print("- Programming (languages, concepts, algorithms)")
    print("- Chemistry (elements, reactions, concepts)")
    print("- Literature (authors, periods, analysis)")
    print("\nSimply tell me what interests you, or type 'quiz me on [topic]' to start a quiz right away.")
    print("Type 'exit' to end the conversation, 'reset' to start over.\n")

    is_complete = False
    topic_selected = False
    selected_topic = ""
    
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
            student_tracker.reset()
            current_state = TutorState.CHAT
            topic_selected = False
            selected_topic = ""
            print("[Conversation has been reset]")
            print("\nWhat topic would you like to learn about today?")
            continue
        
        # Set the topic if this is the first message and not a quiz command
        if not topic_selected and not user_input.lower().startswith("quiz me on"):
            selected_topic = user_input
            topic_selected = True
            print(f"\n[Topic selected: {selected_topic}]")
            # Create a prompt to acknowledge the topic and offer help
            user_input = f"I'd like to learn about {selected_topic}. Can you help me with that?"
        
        # Check if we need to process a quiz answer
        if current_state == TutorState.QUIZ:
            # Simple answer detection - in a real system, this would be more sophisticated
            is_correct = False
            
            # Extract the last question from chat history if available
            # This is a simplified implementation - in a real system, you'd store questions properly
            last_question = ""
            if hasattr(chat, "_chat_history") and chat._chat_history:
                for msg in reversed(chat._chat_history):
                    if msg.name == TUTOR_NAME and "?" in msg.content:
                        last_question = msg.content
                        break
            
            # Process the answer - this is simplified and would need enhancement in a real system
            if "option a" in user_input.lower() or "option b" in user_input.lower() or \
               "option c" in user_input.lower() or "option d" in user_input.lower():
                # Here we'd normally check against the correct answer
                # For demo, we'll simulate some wrong answers about terriers
                if "terrier" in last_question.lower() and not "option c" in user_input.lower():
                    is_correct = False
                else:
                    # Random correctness for demo purposes - replace with actual logic
                    is_correct = "correct" in user_input.lower() or len(user_input) % 2 == 0
                
                # Track the answer
                student_tracker.add_answer(user_input, is_correct)
        
        # Add context variables for selection function
        context_vars = {
            "current_state": current_state.value,
            "consecutive_wrong": student_tracker.consecutive_wrong,
            "problem_topics": ", ".join(student_tracker.get_problem_topics()),
            "chat_history": str(chat._chat_history) if hasattr(chat, "_chat_history") else ""
        }
        
        # First add the user message to chat
        await chat.add_chat_message(message=user_input)
        
        # Then add a system message that contains our context variables
        # We'll format it in a way that the selection function can extract but won't be visible to the user
        context_message = f"[SYSTEM: CONTEXT_VARS current_state={context_vars['current_state']} consecutive_wrong={context_vars['consecutive_wrong']} problem_topics={context_vars['problem_topics']}]"
        
        # Modify the selection function to look for this special context message
        if hasattr(chat.selection_strategy, 'function'):
            chat.selection_strategy.arguments = KernelArguments(
                current_state=context_vars["current_state"],
                consecutive_wrong=context_vars["consecutive_wrong"],
                problem_topics=context_vars["problem_topics"],
                chat_history=context_vars["chat_history"]
            )

        # Look for quiz requests
        if "quiz me on" in user_input.lower() and current_state != TutorState.QUIZ:
            current_state = TutorState.QUIZ
            # Extract topic for quiz
            match = re.search(r"quiz me on\s+(.+)", user_input.lower())
            topic = match.group(1) if match else selected_topic if selected_topic else "general knowledge"
            selected_topic = topic
            topic_selected = True
            
            # Reset the student tracker for new quiz
            student_tracker.reset()

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
            
            # Check if we need to trigger a review based on performance
            if current_state == TutorState.QUIZ and student_tracker.should_trigger_review():
                # In a production system, you'd want to add a message to the chat that triggers the reasoning agent
                print("\n\n[System: Detected pattern of errors. Initiating learning gap analysis...]")
                review_request = f"I notice the student has {student_tracker.consecutive_wrong} consecutive wrong answers or is struggling with specific topics. Please analyze the student's performance and identify potential knowledge gaps."
                await chat.add_chat_message(message=review_request, role="system")
                current_state = TutorState.REVIEW
                    
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
