import os
import logging
logging.basicConfig(level=logging.INFO)

from src.pipeline.nodes.grader import grade_document, grader_node
from src.pipeline.llm_factory import get_llm
from src.pipeline.state import Document

llm = get_llm()

doc: Document = {
    "chunk_id": "test_chunk_1",
    "arxiv_id": "1234",
    "text": "Corrective Retrieval Augmented Generation (CRAG) improves generation robustness.",
    "score": 0.9,
    "title": "Test Paper",
    "authors": [],
    "published": "2024",
    "url": "",
    "abstract": "",
    "grade": "",
    "source": "faiss"
}

print("Testing single document grading...")
res = grade_document("What is CRAG?", doc, llm)
print(res)
