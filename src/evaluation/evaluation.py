import json
import os
import time
import pandas as pd
import sys
from pathlib import Path

from dotenv import load_dotenv
import src.config as config

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parents[2]))

load_dotenv()


from src.pipeline.graph import run_query as run_crag_query
from src.evaluation.graph_naive import run_naive_query

# =====================================================================
#  [CONTROL PANEL] PACING FOR GEMINI FLASH-LITE FREE TIER
# =====================================================================
NUM_QUESTIONS  = 50     # Set to None to run all questions
VERIFY_CSV     = False  # Set to True to skip pipeline and verify CSV generation only
REQUEST_DELAY  = 10     # 60s window safely buffers the 15 RPM API rule
# =====================================================================

def run_evaluation(mode="crag", judge_model="gemini-3.1-flash-lite"):
    """
    Runs pipeline for a specified mode: 'crag' or 'naive'.
    Processes pipeline loops entirely through Gemini 3.1 Flash-Lite.
    Supports secure incremental auto-resume checkpoints without overwriting data.
    """
    print(f"\n[START] KICKING OFF GENERATION RUN: Mode = {mode.upper()} | Model = {judge_model.upper()}\n" + "="*50)

    if VERIFY_CSV:
        print("[WARNING] [VERIFY_CSV MODE] Pipeline will be skipped. Dummy data will be used to verify CSV generation.\n")

    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY is missing from your .env file!")
        print("Grab a free key from https://aistudio.google.com/")
        return

    # Enable LangSmith Tracing mapped out dynamically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"CRAG_Gen_{mode.upper()}_{judge_model.replace(':', '-')}"

    dataset_path = config.PROCESSED_DIR / "eval_dataset.json"
    safe_file_name = judge_model.replace(":", "-").replace(".", "-")

    if VERIFY_CSV:
        output_path = config.PROCESSED_DIR / f"evaluation_results_{mode}_{safe_file_name}_verify.csv"
    else: 
        output_path = config.PROCESSED_DIR / f"evaluation_results_{mode}_{safe_file_name}.csv"

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_data = eval_data[:NUM_QUESTIONS] if NUM_QUESTIONS is not None else eval_data
    print(f"Loaded {len(eval_data)} questions for generation.")

    # ── AUTO-RESUME CHECKPOINT LOADING ───────────────────────────────────────
    completed_questions = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if "question" in existing_df.columns:
                completed_questions = set(existing_df["question"].dropna().tolist())
                print(f"[RESUME] Found existing checkpoint — {len(completed_questions)} questions already done, skipping them.\n")
        except Exception:
            print("[WARNING] Could not read existing CSV for resume — starting fresh.\n")

    for i, item in enumerate(eval_data):
        print(f"[{i+1}/{len(eval_data)}] Testing Question: {item['question']}")

        if item["question"] in completed_questions:
            print(f"    [SKIP] Already processed, skipping.")
            continue

        # ── VERIFY_CSV MODE ──────────────────────────────────────────────────
        if VERIFY_CSV:
            row = {
                "question":         item["question"],
                "answer":           "DUMMY ANSWER - CSV verification mode",
                "ground_truth":     item.get("ground_truth", "N/A"),
                "contexts":         str(["DUMMY CONTEXT"]),
                "pipeline_source": "dummy",
                "web_search_used": False,
                "retry_count":      0,
                "pipeline_grade":   "ungraded",
                "judge_model":      judge_model,
                "retrieval_ms":     0.0,
                "generation_ms":    0.0,
                "total_latency_ms": 0.0,
            }
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
            print(f"    [OK] Dummy row saved to CSV.")
            continue

        # ── NORMAL PIPELINE ROUTING VIA GEMINI FACTORY ───────────────────────
        try:
            import time
            start_time = time.perf_counter()
            if mode == "crag":
                result_state = run_crag_query(item["question"])
            elif mode == "naive":
                result_state = run_naive_query(item["question"])
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            generated_answer   = result_state.get("generation", "Error: No answer generated.")
            retrieved_docs     = result_state.get("documents", [])
            retrieved_contexts = [doc.get("text", "") for doc in retrieved_docs]

            row = {
                "question":         item["question"],
                "answer":           generated_answer,
                "ground_truth":     item["ground_truth"],
                "contexts":         str(retrieved_contexts),
                "pipeline_source":  result_state.get("source", "unknown"),
                "web_search_used":  result_state.get("web_search_used", False),
                "retry_count":      result_state.get("retry_count", 0),
                "pipeline_grade":   result_state.get("grade", "unknown"),
                "judge_model":      judge_model,
                "retrieval_ms":     result_state.get("retrieval_ms", 0.0),
                "generation_ms":    result_state.get("generation_ms", 0.0),
                "total_latency_ms": total_latency_ms,
            }

            # Safely log data incrementally
            pd.DataFrame([row]).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

            print(f"    [OK] Answered and saved! (Source: {row['pipeline_source']} | Retries: {row['retry_count']} | Latency: {total_latency_ms:.1f}ms)")

            # Crucial Sleep Delay to cool down Gemini's free tier RPM limit
            if i < len(eval_data) - 1:
                print(f"    ⏳ Cooling down for {REQUEST_DELAY}s to satisfy Gemini API allocations...")
                time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"    [ERROR] Pipeline failed on this question: {e}")
            
            # Still force a cool down even on error to prevent cascading 429 loops
            if i < len(eval_data) - 1:
                time.sleep(REQUEST_DELAY)
            continue

    # ── Clean Exit ──────────────────────────────────────────────────────────
    print(f"\n[SUCCESS] Generation loop complete!")
    print(f"Your raw dataset is safely preserved inside: {output_path}")

    # Latency Stats Summary
    try:
        df_res = pd.read_csv(output_path)
        if all(col in df_res.columns for col in ["retrieval_ms", "generation_ms", "total_latency_ms"]):
            print(f"\n=== LATENCY SUMMARY STATS ({mode.upper()}) ===")
            for metric in ["retrieval_ms", "generation_ms", "total_latency_ms"]:
                mean_val = df_res[metric].mean()
                p95_val = df_res[metric].quantile(0.95)
                print(f"  {metric:<18} | Mean: {mean_val:8.2f} ms | P95: {p95_val:8.2f} ms")
            print("======================================\n")
    except Exception as e:
        print(f"[WARNING] Could not generate latency summary stats: {e}")

    print("Ready for Step 2: Run your standalone RAGAS script on this CSV.\n")


if __name__ == "__main__":
    run_evaluation(mode="crag", judge_model="gemini-3.1-flash-lite")