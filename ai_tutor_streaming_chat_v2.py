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
It creates a chat interface where you can interact with a Tutor and an Evaluator 
agent that work together to provide educational assistance with pattern detection.
"""

# Define agent names
TUTOR_NAME = "Tutor"
EVALUATOR_NAME = "Evaluator" 
QUIZ_CREATOR_NAME = "QuizCreator"
RESOURCE_CURATOR_NAME = "ResourceCurator"

# Define termination keyword
termination_keyword = "complete"

# Define states for the tutor system
class TutorState(Enum):
    CHAT = "chat"
    QUIZ = "quiz"
    REVIEW = "review"
    RESOURCE = "resource"
    EVALUATION = "evaluation"

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
    
    # Create Tutor agent (focused on conversation)
    tutor_agent = ChatCompletionAgent(
        kernel=kernel,
        name=TUTOR_NAME,
        instructions="""
You are an intelligent tutor named AI Tutor. You are the ONLY agent that directly communicates with the student.

Your responsibilities:
1. Be the primary and ONLY interface between the system and the student
2. Present questions from the QuizCreator to the student EXACTLY as written
3. Deliver evaluations from the Evaluator to the student
4. Share resources from the ResourceCurator with the student
5. Maintain a friendly, encouraging, and educational tone throughout
6. Acknowledge correct answers with positive feedback
7. For incorrect answers, provide the correct answer and brief explanation

When working with other agents:
- When the QuizCreator generates a question, copy and paste it EXACTLY to the student - DO NOT rewrite it or create your own question
- Simply add a brief introduction like "Here's your next question:" before presenting the QuizCreator's question
- When the Evaluator identifies knowledge gaps, explain these insights to the student in a supportive way
- When the ResourceCurator provides learning materials, present them to the student in a structured format

CRITICAL QUIZ INSTRUCTIONS:
- NEVER create your own quiz questions - ONLY copy and present the QuizCreator's questions verbatim
- Always use multiple choice format from QuizCreator with options A, B, C, and D
- After 3 consecutive wrong answers, automatically engage the Evaluator

Remember: You are the ONLY voice the student hears. All other agents communicate through you.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Create Evaluator agent (focused on analyzing performance)
    evaluator_agent = ChatCompletionAgent(
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
After your analysis, ask the Resource Curator to provide appropriate learning materials.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Create Quiz Creator agent (focused on generating quizzes)
    quiz_creator_agent = ChatCompletionAgent(
        kernel=kernel,
        name=QUIZ_CREATOR_NAME,
        instructions="""
You are a specialized quiz creator agent. Your role is to create effective educational multiple-choice quizzes.

Your responsibilities:
1. Generate clear, well-structured multiple-choice quiz questions on the requested topic
2. Always design questions with 4 options (A, B, C, D) per question
3. Ask ONE question at a time - wait for the student to answer before providing the next question
4. Always label each question with its topic/subtopic for categorization
5. Create questions that test understanding, not just memorization
6. Include a mix of difficulty levels to gauge comprehension depth
7. For students who are struggling, create simpler questions on foundational concepts
8. For advanced students, gradually increase difficulty
9. When focusing on problem areas, create targeted questions on those specific topics
10. Store the correct answer for each question for evaluation

Example question format:
Topic: [Specific Topic]
Question: [Clear, well-formed question]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]

IMPORTANT: DO NOT directly interact with the student. Provide your questions to the Tutor agent who will present them to the student.
After generating a question, hand control to the Tutor agent who will present it to the student.
DO NOT ask the student what type of questions they want or any other interaction questions.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Create Resource Curator agent (focused on providing learning materials)
    resource_curator_agent = ChatCompletionAgent(
        kernel=kernel,
        name=RESOURCE_CURATOR_NAME,
        instructions="""
You are a specialized resource curator agent. Your role is to provide tailored learning resources to address identified knowledge gaps.

Your responsibilities:
1. Based on the Evaluator's analysis, identify the most relevant learning materials
2. Provide clear, concise explanations of concepts the student is struggling with
3. Present information in multiple formats when appropriate (descriptions, examples, analogies)
4. Adapt explanations to the student's demonstrated level of understanding
5. Highlight key differences in commonly confused concepts
6. Use concrete examples to illustrate abstract ideas
7. Provide simple exercises or thought experiments to reinforce understanding
8. Connect new information to concepts the student already understands
9. Organize information logically, building from foundational to advanced concepts
10. After providing resources, recommend targeted practice with the Quiz Creator

Sample resource provision format:
CONCEPT: [Brief name of concept]
EXPLANATION: [Clear, concise explanation]
KEY POINTS:
- [Important point 1]
- [Important point 2]
- [Important point 3]
EXAMPLE: [Concrete example showing the concept in action]
COMMON MISCONCEPTION: [Address specific confusion the student demonstrated]

When you've provided appropriate resources, hand back to the Tutor agent.
""",
        function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
    )

    # Define selection function with simpler logic for four agents
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE, CONSECUTIVE WRONG ANSWERS, and CURRENT STATE to choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {TUTOR_NAME} (main conversation agent - ONLY agent that speaks directly to the user)
- {EVALUATOR_NAME} (analyzes performance when student struggles)
- {QUIZ_CREATOR_NAME} (creates multiple-choice questions)
- {RESOURCE_CURATOR_NAME} (provides learning materials after evaluation)

KEY SELECTION RULES:
1. If RESPONSE is from user input:
   - If CURRENT STATE is "quiz", select {QUIZ_CREATOR_NAME} to generate a question
   - Otherwise, select {TUTOR_NAME}

2. Critical agent flow rules:
   - If RESPONSE is from {QUIZ_CREATOR_NAME}, ALWAYS select {TUTOR_NAME} next to present the question
   - If RESPONSE is from {EVALUATOR_NAME}, select {RESOURCE_CURATOR_NAME} next
   - If RESPONSE is from {RESOURCE_CURATOR_NAME}, select {TUTOR_NAME} next
   - If RESPONSE is from {TUTOR_NAME} and contains "quiz" or "test", select {QUIZ_CREATOR_NAME} next

3. Evaluation triggers:
   - If CURRENT STATE is "quiz" AND CONSECUTIVE WRONG ANSWERS is 3+, select {EVALUATOR_NAME}

CURRENT STATE: {{{{$current_state}}}}
CONSECUTIVE WRONG ANSWERS: {{{{$consecutive_wrong}}}}
PROBLEM TOPICS: {{{{$problem_topics}}}}

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define termination function for the four-agent system - simplified to prevent loops
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the current RESPONSE and determine if the conversation should continue or terminate.
Reply only with: {termination_keyword} or continue

STRICT RULES TO PREVENT ENDLESS AGENT DISCUSSIONS:
1. If the Tutor has responded to the user's input: {termination_keyword}
2. If any agent other than Tutor has spoken: continue (so Tutor can relay to the user)

Agent-specific rules:
- If RESPONSE appears to be from {QUIZ_CREATOR_NAME}: continue (so Tutor can present it)
- If RESPONSE appears to be from {EVALUATOR_NAME}: continue (so ResourceCurator can provide materials)
- If RESPONSE appears to be from {RESOURCE_CURATOR_NAME}: continue (so Tutor can present them)
- If RESPONSE appears to be from {TUTOR_NAME} and is a complete thought: {termination_keyword}

REMEMBER: The goal is ONE meaningful agent interaction per user input, not lengthy agent discussions.

CURRENT STATE: {{{{$current_state}}}}
CONSECUTIVE WRONG ANSWERS: {{{{$consecutive_wrong}}}}
PROBLEM TOPICS: {{{{$problem_topics}}}}

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=10)

    # Create the agent group chat with enhanced configuration
    chat = AgentGroupChat(
        agents=[tutor_agent, evaluator_agent, quiz_creator_agent, resource_curator_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=tutor_agent,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value and result.value[0] else TUTOR_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[tutor_agent, quiz_creator_agent, resource_curator_agent, evaluator_agent],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            # Reduced maximum iterations to prevent self-talk loops
            maximum_iterations=2,
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
        
        # Update the arguments for the selection function properly
        # This fixes the "Variable not found in the KernelArguments" error
        if hasattr(chat.selection_strategy, 'function'):
            # Create a new KernelArguments object with our context variables
            args = KernelArguments()
            # Add each key-value pair individually to avoid variable prefix issues
            args["current_state"] = context_vars["current_state"]
            args["consecutive_wrong"] = context_vars["consecutive_wrong"]
            args["problem_topics"] = context_vars["problem_topics"]
            args["chat_history"] = context_vars["chat_history"]
            # Assign the arguments to the selection strategy
            chat.selection_strategy.arguments = args
            
            # Also update the termination strategy arguments if it exists
            if hasattr(chat, 'termination_strategy') and hasattr(chat.termination_strategy, 'function'):
                chat.termination_strategy.arguments = args

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
                    
                    # Capture state transitions based on agent responses
                    if response.name == QUIZ_CREATOR_NAME:
                        current_state = TutorState.QUIZ
                    
                    elif response.name == EVALUATOR_NAME:
                        current_state = TutorState.EVALUATION
                        
                    elif response.name == RESOURCE_CURATOR_NAME:
                        current_state = TutorState.RESOURCE
                        
                    elif response.name == TUTOR_NAME:
                        # Check if we're transitioning out of another state
                        if current_state != TutorState.CHAT and "let's continue" in response.content.lower():
                            current_state = TutorState.CHAT
            
            # Performance evaluation based on the current state
            if current_state == TutorState.QUIZ and student_tracker.should_trigger_review():
                print("\n\n[System: Detected pattern of errors. Initiating learning gap analysis...]")
                review_request = f"@{EVALUATOR_NAME} I notice the student has {student_tracker.consecutive_wrong} consecutive wrong answers or is struggling with specific topics. Please analyze the student's performance and identify potential knowledge gaps."
                await chat.add_chat_message(message=review_request, role="system")
                current_state = TutorState.EVALUATION
            
            # After running a quiz for several questions, request an evaluation
            if current_state == TutorState.QUIZ and len(student_tracker.questions) >= 5 and len(student_tracker.is_correct) >= 3:
                print("\n\n[System: Sufficient questions answered. Requesting evaluation...]")
                
                # Get performance summary
                summary = student_tracker.get_performance_summary()
                
                # Create evaluation request with performance details
                eval_request = f"""
@{EVALUATOR_NAME} Please analyze the student's quiz performance:
- Total questions: {summary['total_questions']}
- Correct answers: {summary['total_correct']}
- Accuracy rate: {summary['accuracy'] * 100:.1f}%
- Consecutive wrong answers: {summary['consecutive_wrong']}
- Problem topics: {', '.join(summary['problem_topics']) if summary['problem_topics'] else 'None identified yet'}
                
Please identify any knowledge gaps or patterns in the student's understanding.
"""
                await chat.add_chat_message(message=eval_request, role="system")
                current_state = TutorState.EVALUATION
                    
        except Exception as e:
            if "Chat is already complete" in str(e):
                # This is normal and expected when conversation turns end
                pass
            else:
                print(f"\nError during chat: {e}")
        
        # Clear the completion flag for the next round
        chat.is_complete = False

if __name__ == "__main__":
    asyncio.run(main())
