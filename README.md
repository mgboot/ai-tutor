# ai-tutor
AI Tutor POC that focuses on identifying the source of student misunderstanding 

AI Tutor is a proof-of-concept system that intelligently identifies gaps in student understanding and provides targeted learning resources. Unlike traditional educational approaches that follow a fixed curriculum, this system dynamically assesses student comprehension, pinpoints specific knowledge deficiencies, and delivers customized materials to address those gaps. 

The AI Tutor consists of four core components:
1. **Quiz Creator** - Generates assessments to evaluate student understanding
2. **Evaluator** - Analyzes responses to identify specific knowledge gaps
3. **Resource Curator** - Provides tailored learning materials to address deficiencies
4. **Conversational Interface** - Enables students to chat directly with the AI for questions, clarification, and support

By focusing on the precise source of student misunderstanding rather than simply noting incorrect answers, the system ensures learners build knowledge on solid foundations before progressing to more advanced concepts.


# Example
	• Student is studying electronics and circuit design.
	• At some chapter in the book the AI Tutor (Quiz Creator) will create a test to see if they understand the concepts and can do exercises commensurate with what they have been studying.
	• The AI tutor (evaluator) notices after several questions that the student seems to always get stuck on parts that involve capacitors.  They understand resistors, transistors, etc. really well, but get the final answer wrong based on this lack of knowledge in a base skill like capacitors.  
	• The AI Tutor (Resource Curator) presents the student with a couple chapters or paragraphs explaining capacitors again, or from a different text book (or online video).
	• The AI Tutor (Quiz Creator) then builds new test to see if the student now has a solid understanding of capacitors and will not be "left behind without a base understanding" as the class moves on.  
	• This system ensures the student doesn't sit in a class/lecture not really understanding some basic pieces and waste time when they could be evaluated and redirected to what will help them most right now.  Once they have that they will more quickly catch back up with the class and be able to take advantage of the new learnings with a solid base.
![image](https://github.com/user-attachments/assets/49b2d5fe-ecff-498d-a243-39635d4adec9)

## Another Example
* Student is studying canine breeds and characteristics.
* The AI Tutor (Quiz Creator) generates a multiple choice quiz with questions about different dog breed groups, physical traits, and origins.
* The AI tutor (Evaluator) identifies that the student consistently misidentifies terrier breeds, confusing them with sporting dogs, while correctly answering questions about hounds, working dogs, and toy breeds.
* The AI Tutor (Resource Curator) provides the student with targeted materials about terrier characteristics, history, and distinctive features, including visual examples comparing terriers to sporting breeds.
* The AI Tutor (Quiz Creator) then develops a focused follow-up quiz specifically on terrier identification and classification to ensure the knowledge gap is addressed.
* This approach prevents the student from proceeding with a fundamental misunderstanding that would impair their ability to correctly classify breeds in more advanced contexts, ensuring they build knowledge on a solid foundation.