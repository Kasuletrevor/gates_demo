"""
GatesGPT QA API Server.
"""

from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
from question_answering.utils import init_vector_database
from question_answering.utils import ask_a_question

class Question(BaseModel):
    question: str
    language: str
    category: str
    sub_category: Union[str, None]
    topic: Union[str, None]
    sub_topic: Union[str, None]
    location: Union[str, None]

# Metadata
description = """
GatesGPTAPI is a Question Answering Platform for agricultural questions.

## Supported features

- Querying the API with a question and receive an answer.
- Updating question answer vector database with new information (_not implemented_).
"""
app = FastAPI(
    title="GatesGPTAPI",
    description=description,
    version="0.0.1",
)

# OPERATIONS
@app.get("/")
async def root():
    return {"Welcome to the GatesGPT API Server"}

@app.post("/query/")
async def query(question: Question):
    results = ask_a_question(question=question.question)
    
    # Process the results to a serializable format if necessary.
    processed_results = {
        "ids": results['ids'],
        "similar_questions": results['similar_questions'],
        "answers": results['answers'],
        "distances": results['distances']
    }

    return processed_results

@app.post("/init")
async def initialize():
    init_vector_database()