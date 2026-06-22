import os
import sys
import pandas as pd
from datasets import Dataset
import ast
from dotenv import load_dotenv

load_dotenv()

# Allow imports from project root
sys.path.append(os.path.abspath("."))

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

# Load CSV
df = pd.read_csv('data/processed/evaluation_results_crag_gemini-3-1-flash-lite.csv')
def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return [val]
df['contexts'] = df['contexts'].apply(safe_eval)

# Take first row
row = df.iloc[0]
data_dict = {
    "question": [row["question"]],
    "answer": [row["answer"]],
    "contexts": [row["contexts"]],
    "ground_truth": [row["ground_truth"]]
}
dataset = Dataset.from_dict(data_dict)

print("\n--- DATASET ROW ---")
print("Question:", data_dict["question"][0])
print("Answer:", data_dict["answer"][0][:100], "...")
print("Contexts count:", len(data_dict["contexts"][0]))
print("Ground Truth:", data_dict["ground_truth"][0][:100], "...")
print("-------------------\n")

from langchain_core.callbacks import BaseCallbackHandler

class GlobalRequestCounter(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.count += 1
        print(f"      [CALLBACK] LLM Call started! Current cumulative count: {self.count}")

global_counter = GlobalRequestCounter()

eval_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    google_api_key=os.environ["GEMINI_API_KEY"],
    callbacks=[global_counter]
)
eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ragas_judge      = LangchainLLMWrapper(eval_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
safe_answer_relevancy = AnswerRelevancy(strictness=1)

print("Running evaluate with raise_exceptions=True...")
try:
    res = evaluate(
        dataset=dataset,
        metrics=[faithfulness, safe_answer_relevancy, context_precision],
        llm=ragas_judge,
        embeddings=ragas_embeddings,
        raise_exceptions=True
    )
    print("\nSUCCESS!")
    print("Result:", res)
    df_res = res.to_pandas()
    import json
    print("DataFrame Columns:", df_res.columns.tolist())
    print("DataFrame Row Dict:")
    row_dict = df_res.to_dict(orient="records")[0]
    short_dict = {}
    for k, v in row_dict.items():
        if isinstance(v, str) and len(v) > 200:
            short_dict[k] = v[:200] + "..."
        elif isinstance(v, list):
            short_dict[k] = [x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x for x in v]
        else:
            short_dict[k] = v
    print(f"\nFinal global counter count: {global_counter.count}")
except Exception as e:
    print(f"\nFinal global counter count before failure: {global_counter.count}")
    print("\nFAILED with exception:")
    import traceback
    traceback.print_exc()
