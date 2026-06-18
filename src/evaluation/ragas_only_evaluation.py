import os
import time
import pandas as pd
import types
import sys
import ast
from datasets import Dataset
from dotenv import load_dotenv

# =====================================================================
# 🚨 SURGICAL PATCH FOR RAGAS INTERNAL VERTEX BUG 🚨
dummy_module = types.ModuleType('langchain_community.chat_models.vertexai')
class DummyVertexAI: pass
dummy_module.ChatVertexAI = DummyVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = dummy_module
# =====================================================================

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Project's original imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def evaluate_existing_csv(mode="naive"):
    # 1. Path to your generated CSV file
    csv_path = f"data/processed/evaluation_results_{mode}_gemini-3-1-flash-lite.csv"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find the CSV file at {csv_path}")
        return

    print(f"[1/4] Loading generated CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Clean and format the data for RAGAS
    print("[2/4] Formatting data fields for RAGAS evaluation...")
    
    def safe_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]

    df['contexts'] = df['contexts'].apply(safe_eval)

    data_dict = {
        "question":     df["question"].tolist(),
        "answer":       df["answer"].tolist(),
        "contexts":     df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist()
    }
    dataset = Dataset.from_dict(data_dict)

    # 3. Initialize the Gemini 3.1 Flash-Lite Judge
    ragas_model_name = "gemini-3.1-flash-lite"
    print(f"[3/4] Initializing {ragas_model_name} as the RAGAS evaluation judge...")
    
    # Clean, standard initialization
    eval_llm = ChatGoogleGenerativeAI(
        model=ragas_model_name,
        temperature=0.0,
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ragas_judge      = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # =================================================================
    # 🚨 THE FIX: Force Answer Relevancy to strictly request 1 candidate
    # =================================================================
    safe_answer_relevancy = AnswerRelevancy(strictness=1)

    # 4. Compute Scores Safely (Paced to respect rate limits)
    print(f"[4/4] Launching RAGAS evaluation suite... Pacing for 15 RPM limit.")
    print("="*60)
    
    results_list = []
    
    for i in range(len(dataset)):
        print(f"Grading Question [{i+1}/{len(dataset)}]...")
        
        single_row = dataset.select([i])
        
        try:
            row_result = evaluate(
                dataset=single_row,
                metrics=[faithfulness, safe_answer_relevancy], # Using the configured metric
                llm=ragas_judge,
                embeddings=ragas_embeddings,
                raise_exceptions=False 
            )
            
            row_df = row_result.to_pandas()
            results_list.append({
                "faithfulness": row_df["faithfulness"].iloc[0],
                "answer_relevancy": row_df["answer_relevancy"].iloc[0]
            })
            
        except Exception as e:
            print(f"  [ERROR] Skipping row {i+1} due to API error: {e}")
            results_list.append({"faithfulness": None, "answer_relevancy": None})
            
        # ⏳ Safe 15-Second Delay
        if i < len(dataset) - 1:
            time.sleep(15) 

    # 5. Export final results
    print("\n" + "="*60)
    print("[MERGE] Integrating newly graded questions back into previous tracking logs...")
    
    scores_df = pd.DataFrame(results_list)
    df["faithfulness"] = scores_df["faithfulness"]
    df["answer_relevancy"] = scores_df["answer_relevancy"]
    df["ragas_judge_model"] = ragas_model_name

    df.to_csv(csv_path, index=False)
    
    print("[SUCCESS] RAGAS Metrics calculated and merged successfully!")
    print(f"Final benchmark reports saved cleanly inside: {csv_path}\n")

if __name__ == "__main__":
    evaluate_existing_csv(mode ="naive")