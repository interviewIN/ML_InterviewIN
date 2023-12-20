from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.analyzer import HRInterviewAnalyzer

# Additional comments and documentation
"""
This FastAPI app defines an endpoint '/get_summary' that analyzes HR interview data and returns a summary.

1. QNA model (pydantic.BaseModel):
    - Defines the structure of a single question and answer pair.

2. HRInterviewInput model (pydantic.BaseModel):
    - Defines the expected structure of the incoming JSON data for the '/get_summary' endpoint.
    - Includes candidate_name, job_title, company_name, and a list of QNA pairs.

3. '/get_summary' endpoint (POST):
    - Accepts HRInterviewInput data.
    - Extracts questions and answers from input_data.
    - Calls HRInterviewAnalyzer to analyze the interview.
    - Returns the analysis result as a dictionary.

4. Exception handling:
    - If an exception occurs during analysis, raises an HTTPException with a 500 status code.
"""

# To run the FastAPI app, use the following command:
# uvicorn filename:app --reload
# Replace 'filename' with the actual filename containing the FastAPI app.

app = FastAPI()
analyzer = HRInterviewAnalyzer()

class QNA(BaseModel):
    question: str
    answer: str

class HRInterviewInput(BaseModel):
    candidate_name: str
    job_title: str
    company_name: str
    interview_qna: list[QNA]

@app.post("/get_summary")
async def get_summary(input_data: HRInterviewInput):
    try:
        # Extract questions and answers from input_data
        interview_qna_list = [
            {"question": item.question, "answer": item.answer}
            for item in input_data.interview_qna
        ]

        # Analyze the interview using HRInterviewAnalyzer
        result = analyzer.analyze_interview(
            job_title=input_data.job_title,
            company_name=input_data.company_name,
            candidate_name=input_data.candidate_name,
            interview_qna=interview_qna_list
        )

        # Return the result as a dictionary
        return result.dict()
    except Exception as e:
        # If an exception occurs during analysis, raise an HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=str(e))
