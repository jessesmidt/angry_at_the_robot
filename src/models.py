from pydantic import BaseModel, Field
from enum import Enum
import uuid

# The following pydantic models for type-safe data handling must be implemented. These
# models ensure data integrity and provide automatic validation throughout the pipeline.
# The MinimalSource model represents a minimal source of information:
# MinimalSource Model

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

# The UnansweredQuestion and AnsweredQuestion models represent an unan-
# swered question and an answered question:
# UnansweredQuestion and AnsweredQuestion Models

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
    str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str

# The RagDataset model represents a dataset of RAG questions:
# RagDataset Model

class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

# The MinimalSearchResults and MinimalAnswer models represent the search
# results and an answer:
# MinimalSearchResults and MinimalAnswer Models

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str
    
# The StudentSearchResults and StudentSearchResultsAndAnswer models rep-
# resent search results and search results with answers:
# StudentSearchResults and StudentSearchResultsAndAnswer Models

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]

# The provided models are a foundation. You can expand them by adding new models
# or extra fields (for example in the search results model) if your implementation requires
# it.