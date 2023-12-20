from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class QNASummary(BaseModel):
    """
    Pydantic model representing the structure of the output summary.

    Attributes:
    - name: Name of the candidate.
    - overall_impression: Brief description of the candidate's demeanor, communication skills, and presentation.
    - chance_of_getting_the_job: Assessment of the candidate's suitability for the position.
    - most_relevant_position: Suggested position based on skills and interview discussion.
    - personal_capability: Summary of the candidate's soft skills (teamwork, communication, leadership, etc.).
    - psychological_capability: Evaluation of stress tolerance, adaptability, and emotional intelligence.
    - technical_capability: Assessment of technical skills and knowledge.
    - final_thoughts: Overall impression of the candidate and fit within the company culture.
    """
    name: str = Field(description="name of the candidate")
    overall_impression: str = Field(description="Briefly describe the candidate's general demeanor, communication skills, and presentation during the interview.")
    chance_of_getting_the_job: str = Field(description="Assess the candidate's overall suitability for the position based on their qualifications, experience, and interview performance.")
    most_relevant_position: str = Field(description="If the candidate applied for multiple positions, suggest the most suitable one based on their skills and the interview discussion.")
    personal_capability: str = Field(description="Summarize the candidate's soft skills, such as teamwork, communication, leadership, and problem-solving abilities, as demonstrated during the interview.")
    psychological_capability: str = Field(description="Evaluate the candidate's stress tolerance, adaptability, and emotional intelligence based on their responses and overall demeanor.")
    technical_capability: str = Field(description="Assess the candidate's technical skills and knowledge relevant to the specific job requirements. Briefly summarize their answers to technical questions and highlight areas of expertise or potential gaps.")
    final_thoughts: str = Field(description="Briefly summarize your overall impression of the candidate and their potential fit within the company culture. Provide any concluding remarks or recommendations for further evaluation if needed.")

class HRInterviewAnalyzer:
    """
    Class for analyzing HR interview transcripts and generating summaries.

    Attributes:
    - llm: Language Model used for generating summaries.
    - parser: PydanticOutputParser for parsing model outputs into QNASummary objects.
    - hr_template: Template for HR interview transcripts.
    - prompt_template: PromptTemplate for formatting input data.

    Methods:
    - analyze_interview: Analyzes an HR interview transcript and returns a summary.
    """
    def __init__(self, model_name="code-bison", max_output_tokens=1000, temperature=0.3):
        """
        Initializes the HRInterviewAnalyzer.

        Args:
        - model_name: Name of the language model.
        - max_output_tokens: Maximum number of output tokens.
        - temperature: Model sampling temperature.
        """
        self.llm = VertexAI(model_name=model_name, max_output_tokens=max_output_tokens, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=QNASummary)
        # Define HR interview transcript template
        self.hr_template = """
            Here is a transcript of an HR interview for the {job_title} position at {company_name}. The candidate is {candidate_name}. Please analyze the interview and provide a bulleted summary with the following sections:
            {format_instructions}
            ---

            {interview_qna}

            ---
            """

        # Define PromptTemplate object
        self.prompt_template = PromptTemplate(
            input_variables=["job_title", "company_name", "candidate_name", "interview_qna"],
            template=self.hr_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def analyze_interview(self, job_title, company_name, candidate_name, interview_qna):
        """
        Analyzes an HR interview transcript and returns a summary.

        Args:
        - job_title: The job title for which the interview was conducted.
        - company_name: The name of the company conducting the interview.
        - candidate_name: The name of the candidate being interviewed.
        - interview_qna: The transcript of the interview, including questions and answers.

        Returns:
        - A dict containing the parsed analysis of the interview.
        """
        input_data = self.prompt_template.format(
            job_title=job_title,
            company_name=company_name,
            candidate_name=candidate_name,
            interview_qna=interview_qna,
        )
        return self.parser.parse(self.llm(input_data))

# example_qna = """
# 1. Question: Can you explain the difference between procedural programming and object-oriented programming (OOP)?

# Answer:
# Procedural programming is a programming paradigm that focuses on procedures or routines, where the program is structured in a linear manner. It uses functions to perform tasks and often relies on global variables. On the other hand, Object-Oriented Programming (OOP) is a paradigm that uses objects, which encapsulate data and behavior. It promotes concepts like encapsulation, inheritance, and polymorphism for more modular and reusable code.

# 2. Question: What is the significance of version control, and how do you use it in your development workflow?

# Answer:
# Version control is crucial for tracking changes in software development, enabling collaboration, and reverting to previous states if needed. I commonly use Git for version control. I create branches for new features or bug fixes, regularly commit changes with descriptive messages, and merge branches back into the main branch when features are complete and tested.

# 3. Question: How do you ensure the security of a web application you're developing?

# Answer:
# Security is paramount in web development. I follow best practices such as input validation, parameterized queries to prevent SQL injection, and use of secure communication protocols like HTTPS. I implement proper authentication and authorization mechanisms, regularly update dependencies, and conduct security audits. Additionally, I stay informed about the latest security threats and incorporate security measures accordingly.

# 4. Question: Describe the process of optimizing code for performance.

# Answer:
# Code optimization is essential for improving performance. I start by profiling the code to identify bottlenecks. Once identified, I focus on algorithmic improvements, use of efficient data structures, and minimizing I/O operations. I also pay attention to memory usage and cache efficiency. Regular testing and benchmarking help ensure that optimizations don't compromise code readability or introduce new bugs.

# 5. Question: How do you stay updated with the latest developments and trends in the software engineering industry?

# Answer:
# Staying current is crucial in this rapidly evolving field. I regularly read tech blogs, follow reputable industry websites, and participate in online communities like GitHub and Stack Overflow. Attending conferences, webinars, and meetups provides opportunities to learn from experts and engage with the broader software engineering community. I also experiment with new technologies through personal projects to gain hands-on experience.
# """

# # Example usage
# analyzer = HRInterviewAnalyzer()
# summary = analyzer.analyze_interview(
#     job_title="software engineer",
#     company_name="google",
#     candidate_name="John Doe",
#     interview_qna=example_qna # Insert actual interview transcript here
# )

# print(summary)
# print(summary.overall_impression)