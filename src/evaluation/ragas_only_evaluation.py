import os
import time
import pandas as pd
import types
import sys
import ast
from datasets import Dataset
from dotenv import load_dotenv
import src.config as config

# =====================================================================
# 🚨 SURGICAL PATCH FOR RAGAS INTERNAL VERTEX BUG 🚨
dummy_module = types.ModuleType('langchain_community.chat_models.vertexai')
class DummyVertexAI: pass
dummy_module.ChatVertexAI = DummyVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = dummy_module
# =====================================================================

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Project's original imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

from langchain_core.callbacks import BaseCallbackHandler

class GlobalRequestCounter(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.count += 1

def evaluate_existing_csv(mode="naive", run_only_index=None):
    # 1. Path to your generated CSV file
    csv_path = config.PROCESSED_DIR / f"evaluation_results_{mode}_gemini-3-1-flash-lite.csv"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find the CSV file at {csv_path}")
        return

    print(f"\n[START] READING CSV FROM PATH: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)

    # Initialize RAGAS columns if not present
    for col in ["faithfulness", "answer_relevancy", "context_precision", "ragas_judge_model"]:
        if col not in df.columns:
            df[col] = None

    # 2. Clean and format the data for RAGAS
    print("[2/4] Formatting data fields for RAGAS evaluation...")
    
    def safe_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]

    # Parse contexts for dataset, but do not overwrite df['contexts'] as list object in-place to avoid saving issues
    parsed_contexts = df['contexts'].apply(safe_eval).tolist()

    data_dict = {
        "question":     df["question"].tolist(),
        "answer":       df["answer"].tolist(),
        "contexts":     parsed_contexts,
        "ground_truth": df["ground_truth"].tolist()
    }
    dataset = Dataset.from_dict(data_dict)

    # 3. Initialize model settings
    ragas_model_name = "gemini-3.1-flash-lite"
    safe_answer_relevancy = AnswerRelevancy(strictness=1)
    
    global_counter = GlobalRequestCounter()

    import types
    # Dynamically override context_precision's get_row_attributes to truncate contexts to top-3
    # This keeps a single clean evaluate call and avoids closed event loop / gRPC channel errors
    def custom_get_row_attributes(self, row):
        return row["user_input"], row["retrieved_contexts"][:3], row["reference"]

    context_precision._get_row_attributes = types.MethodType(custom_get_row_attributes, context_precision)

    # 4. Compute Scores Safely (Paced to respect rate limits, with incremental saving and 429 prevention)
    print(f"[4/4] Launching RAGAS evaluation suite... Pacing for 15 RPM limit.")
    print("="*60)
    
    for i in range(len(dataset)):
        # Check if the row already has valid (non-null) scores for all three metrics
        has_faithfulness = pd.notna(df.loc[i, "faithfulness"])
        has_relevancy = pd.notna(df.loc[i, "answer_relevancy"])
        has_precision = pd.notna(df.loc[i, "context_precision"])
        
        # Dynamically build the metrics to evaluate for this row
        metrics_to_run = []
        metric_names_to_run = []
        if not has_faithfulness:
            metrics_to_run.append(faithfulness)
            metric_names_to_run.append("faithfulness")
        if not has_relevancy:
            metrics_to_run.append(safe_answer_relevancy)
            metric_names_to_run.append("answer_relevancy")
        if not has_precision:
            metrics_to_run.append(context_precision)
            metric_names_to_run.append("context_precision")

        if not metrics_to_run:
            print(f"Skipping Question [{i+1}/{len(df)}] - already evaluated (faithfulness={df.loc[i, 'faithfulness']:.3f}, relevancy={df.loc[i, 'answer_relevancy']:.3f}, precision={df.loc[i, 'context_precision']:.3f})")
            if run_only_index is not None and i == run_only_index:
                print(f"Target index {run_only_index} was skipped because it's already evaluated. Stopping dry-run.")
                break
            continue

        # If run_only_index is specified, bypass all other indexes
        if run_only_index is not None and i != run_only_index:
            if i < run_only_index:
                print(f"Bypassing Question [{i+1}/{len(df)}] to reach target index {run_only_index+1}...")
                continue
            else:
                break

        print(f"\nGrading Question [{i+1}/{len(dataset)}] (running metrics: {', '.join(metric_names_to_run)})...")
        single_row = dataset.select([i])
        
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Re-initialize judge and embeddings on the active event loop
            eval_llm = ChatGoogleGenerativeAI(
                model=ragas_model_name,
                temperature=0.0,
                google_api_key=os.environ["GEMINI_API_KEY"],
                callbacks=[global_counter],
                max_retries=0  # Prevent LangChain from retrying quota errors multiple times
            )
            
            eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            ragas_judge      = LangchainLLMWrapper(eval_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

            # Evaluate only the missing metrics in a single call
            row_result = evaluate(
                dataset=single_row,
                metrics=metrics_to_run,
                llm=ragas_judge,
                embeddings=ragas_embeddings,
                raise_exceptions=True
            )
            row_df = row_result.to_pandas()
            
            # Save results immediately to df
            if faithfulness in metrics_to_run:
                df.loc[i, "faithfulness"] = row_df["faithfulness"].iloc[0]
            if safe_answer_relevancy in metrics_to_run:
                df.loc[i, "answer_relevancy"] = row_df["answer_relevancy"].iloc[0]
            if context_precision in metrics_to_run:
                df.loc[i, "context_precision"] = row_df["context_precision"].iloc[0]
            df.loc[i, "ragas_judge_model"] = ragas_model_name
            
            # Write immediately to disk
            df.to_csv(csv_path, index=False)
            print(f"  [OK] Saved results for Question {i+1} to CSV.")
            print(f"  [HTTP] Cumulative Gemini API Requests so far: {global_counter.count}")
            
            # If we did a single-row dry run, stop here
            if run_only_index is not None:
                print(f"[DRY-RUN DONE] Successfully completed single-row evaluation for index {run_only_index}.")
                break
            
        except Exception as e:
            err_str = str(e).lower()
            is_quota_error = any(kw in err_str for kw in ["429", "resourceexhausted", "quota", "rate limit"])
            
            if is_quota_error:
                print(f"\n[QUOTA EXCEEDED] Hit Gemini API rate limit / quota error at Question {i+1}: {e}")
                print(f"Cumulative Gemini API Requests before stopping: {global_counter.count}")
                print("Stopping evaluation loop immediately to preserve remaining quota and avoid API spam.")
                break
            else:
                print(f"  [ERROR] Skipping Question {i+1} due to non-quota error: {e}")
                if faithfulness in metrics_to_run:
                    df.loc[i, "faithfulness"] = None
                if safe_answer_relevancy in metrics_to_run:
                    df.loc[i, "answer_relevancy"] = None
                if context_precision in metrics_to_run:
                    df.loc[i, "context_precision"] = None
                df.loc[i, "ragas_judge_model"] = ragas_model_name
                df.to_csv(csv_path, index=False)
                
        # ⏳ Safe 15-Second Delay between rows
        if i < len(dataset) - 1:
            time.sleep(15)

    print("\n" + "="*60)
    print(f"[FINISHED] Evaluation pass for mode='{mode}' finished or stopped. Results saved inside: {csv_path}\n")

if __name__ == "__main__":
    evaluate_existing_csv(mode="crag")