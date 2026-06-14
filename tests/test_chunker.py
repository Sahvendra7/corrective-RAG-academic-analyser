import re
from pathlib import Path

# --- Testing Configuration ---
# We use smaller sizes here so you can easily read the terminal output
TEST_CHUNK_SIZE = 100       
TEST_CHUNK_OVERLAP = 20     
TEST_MIN_CHUNK_SIZE = 10    

def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences using regex."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, arxiv_id: str) -> list[dict]:
    """Test version of your chunking logic with smaller targets."""
    sentences = split_into_sentences(text)
    chunks = []
    chunk_index = 0
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += sentence_words

        if current_word_count >= TEST_CHUNK_SIZE:
            chunk_text_str = " ".join(current_sentences)
            if len(chunk_text_str.split()) >= TEST_MIN_CHUNK_SIZE:
                chunks.append({
                    "chunk_id": f"{arxiv_id}_chunk_{chunk_index}",
                    "text": chunk_text_str,
                    "word_count": len(chunk_text_str.split()),
                })
                chunk_index += 1

            overlap_sentences = []
            overlap_word_count = 0
            
            # --- THE FIXED OVERLAP LOGIC ---
            for sent in reversed(current_sentences):
                sent_words = len(sent.split())
                overlap_sentences.insert(0, sent)
                overlap_word_count += sent_words
                
                # Break ONLY AFTER we have met or exceeded the overlap quota
                if overlap_word_count >= TEST_CHUNK_OVERLAP:
                    break
            # -------------------------------

            current_sentences = overlap_sentences
            current_word_count = overlap_word_count

    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if len(chunk_text_str.split()) >= TEST_MIN_CHUNK_SIZE:
            chunks.append({
                "chunk_id": f"{arxiv_id}_chunk_{chunk_index}",
                "text": chunk_text_str,
                "word_count": len(chunk_text_str.split()),
            })

    return chunks

def run_test(txt_path: str, arxiv_id: str):
    print(f"--- Testing Chunker on: {arxiv_id} ---")
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find file at {txt_path}")
        return

    chunks = chunk_text(text, arxiv_id)
    
    print(f"Total chunks generated: {len(chunks)}\n")
    
    # Print only the first 2 chunks so we can visually inspect the overlap
    for i in range(min(2, len(chunks))):
        print(f"[{chunks[i]['chunk_id']}] (Words: {chunks[i]['word_count']})")
        print("-" * 40)
        print(chunks[i]['text'])
        print("-" * 40 + "\n")

# --- EXECUTION ---
# Make sure this points to your actual text file!
test_file_path = "data/processed/texts/1007.5016.txt" 
test_arxiv_id = "1007.5016"

run_test(test_file_path, test_arxiv_id)