import os
import json
import random
import time
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types, errors

from dotenv import load_dotenv
import src.config as config

load_dotenv()

# ── 1. Define the Structured Output Schema ────────────────────────────────────
class EvalItem(BaseModel):
    question: str = Field(description="A highly technical question based strictly on the provided context.")
    ground_truth: str = Field(description="The complete, factually accurate answer directly supported by the context.")
    context: str = Field(description="The exact snippet or paragraph from the source text used to answer the question.")
    arxiv_id: str = Field(description="The arXiv ID or paper filename source.")

class EvalDataset(BaseModel):
    items: list[EvalItem]

# ── 2. Main Generation Logic ──────────────────────────────────────────────────
def generate_dataset(num_questions_target: int = 50):
    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] Please set your GEMINI_API_KEY environment variable.")
        return

    client = genai.Client()
    chunks_dir = config.CHUNK_DIR
    output_path = config.PROCESSED_DIR / "eval_dataset.json"
    
    if not chunks_dir.exists():
        print(f"[ERROR] Chunks directory not found at {chunks_dir}.")
        return

    # --- CHECKPOINTING: Load existing progress ---
    all_generated_items = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_generated_items = json.load(f)
            print(f"Found {len(all_generated_items)} existing questions. Resuming progress...")
        except json.JSONDecodeError:
            print("Existing JSON was corrupted. Starting fresh.")
            all_generated_items = []

    if len(all_generated_items) >= num_questions_target:
        print("You already have enough questions in your dataset!")
        return

    chunk_files = list(chunks_dir.glob("*.json"))
    random.shuffle(chunk_files)
    
    print(f"Need to generate {num_questions_target - len(all_generated_items)} more questions...")
    
    consecutive_failures = 0

    for file_path in chunk_files:
        if len(all_generated_items) >= num_questions_target:
            break
            
        if consecutive_failures >= 3:
            print("\n API is consistently blocking us. Pausing execution. Run the script again later to resume.")
            break

        arxiv_id = file_path.stem.replace("_chunks", "")
        success = False
        retries = 0
        
        while not success and retries < 3:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                
                sampled_chunks = random.sample(chunks_data, min(3, len(chunks_data)))
                context_block = "\n\n".join([c.get("text", c.get("content", "")) for c in sampled_chunks if isinstance(c, dict)])
                
                if not context_block.strip():
                    break 
                    
                print(f" -> Generating questions from Paper {arxiv_id}...")
                
                prompt = f"""
                You are building a golden dataset to evaluate an academic QA system.
                Based EXCLUSIVELY on the text block provided below from arXiv paper {arxiv_id}, generate 2 distinct, highly technical questions and their answers.
                
                CRITICAL INSTRUCTIONS FOR QUESTIONS:
                - Make the questions sound like they were typed by a real human researcher asking a chatbot.
                - Keep questions concise (1-2 sentences maximum). 
                - Do NOT ask multi-part exam-style questions.
                - Ask about concepts, mechanisms, or architectures.
                
                Do not make assumptions or extrapolate outside this context.
                
                Context text:
                \"\"\"
                {context_block}
                \"\"\"
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EvalDataset,
                        temperature=0.2
                    ),
                )
                
                batch_data = json.loads(response.text)
                
                for item in batch_data.get("items", []):
                    item["arxiv_id"] = arxiv_id
                    all_generated_items.append(item)
                
                # --- CHECKPOINTING: Save to disk immediately ---
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_generated_items, f, indent=4)
                
                success = True
                consecutive_failures = 0 # Reset failure count
                print(f" Saved! Current total: {len(all_generated_items)}/{num_questions_target}. Sleeping 10s...")
                time.sleep(10)

            except errors.ClientError as e:
                if e.code == 429:
                    print(f" Rate limit hit. Cooling down for 60 seconds...")
                    time.sleep(60)
                    retries += 1
                else:
                    print(f"[WARNING] API Error: {e.message}")
                    break
            except Exception as e:
                print(f"[WARNING] Unhandled error: {e}")
                break
                
        if not success:
            consecutive_failures += 1

    print("\n" + "="*50)
    print(f"🏁 Run finished. Current dataset size: {len(all_generated_items)} questions.")
    print("="*50)

if __name__ == "__main__":
    generate_dataset(num_questions_target=50)