import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Allow imports from project root
sys.path.append(os.path.abspath("."))

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class DocumentGrade(BaseModel):
    grade: str = Field(
        description="Relevance grade: 'relevant', 'ambiguous', or 'irrelevant'"
    )
    reasoning: str = Field(
        description="One sentence explanation of why this grade was assigned"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    google_api_key=os.environ["GEMINI_API_KEY"]
)

# Test 1: Default with_structured_output
structured_llm = llm.with_structured_output(DocumentGrade)

from langchain_core.messages import SystemMessage, HumanMessage
from src.pipeline.nodes.grader import GRADER_SYSTEM_PROMPT, GRADER_HUMAN_PROMPT

# Override prompt to remove json constraints
CLEAN_GRADER_SYSTEM_PROMPT = """You are an expert document relevance grader for a research paper Q&A system.

Your job is to assess whether a retrieved document chunk is relevant to the user's question.

Grade each document as exactly one of:
- "relevant"   : The document directly addresses, answers, or strongly supports the question
- "ambiguous"  : The document is related to the topic but does not directly answer the question
- "irrelevant" : The document has no meaningful connection to the question

Be strict. A document about a related topic but not the specific question is "ambiguous", not "relevant"."""

messages = [
    SystemMessage(content=CLEAN_GRADER_SYSTEM_PROMPT),
    HumanMessage(
        content=GRADER_HUMAN_PROMPT.format(
            query="How does the proposed context-aware cognitive augmentation framework dynamically adjust AI interventions to support cognitive flow?",
            text="Corrective Retrieval Augmented Generation (CRAG) is a framework..."
        )
    )
]

print("Test 1 (Actual Prompt with Clean System Prompt): invoking structured_llm...")
try:
    res = structured_llm.invoke(messages)
    print("Result type:", type(res))
    print("Result:", res)
except Exception as e:
    print("Exception:")
    import traceback
    traceback.print_exc()
