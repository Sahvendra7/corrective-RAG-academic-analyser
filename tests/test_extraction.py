import fitz
from pathlib import Path
import pytest

@pytest.fixture
def pdf_path(tmp_path):
    """Generates a temporary PDF file for testing text extraction."""
    pdf_file = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Insert sample text blocks
    page.insert_text((50, 50), "This is a test block of text on the page.")
    page.insert_text((50, 100), "Another block of text.")
    doc.save(str(pdf_file))
    doc.close()
    return str(pdf_file)

def test_single_page_extraction(pdf_path: str, page_number: int = 0):
    """Tests block-level extraction on a single PDF page."""
    
    print(f"--- Testing Extraction on: {pdf_path} (Page {page_number}) ---")
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        page_rect = page.rect
        header_threshold = page_rect.height * 0.08 
        footer_threshold = page_rect.height * 0.92 

        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        page_content = []
        for b in blocks:
            y0, y1, block_type, text = b[1], b[3], b[6], b[4].strip()

            if block_type != 0:
                continue
            if y0 < header_threshold or y1 > footer_threshold:
                continue
            
            if text:
                page_content.append(text)

        # Print the cleaned, reconstructed text
        final_text = "\n\n".join(page_content)
        print(final_text)
        print(f"\n--- End of Page {page_number} ---")
        print(f"Total Characters Extracted: {len(final_text)}")
        
        doc.close()

    except Exception as e:
        print(f"Failed to extract text: {e}")
        raise e

if __name__ == "__main__":
    # Run the test on the first page (index 0) of a specific PDF
    # Update this string with a real PDF name from your data/raw/ folder!
    test_pdf_path = "/Users/sahvendraz/Desktop/CRAG/data/raw/1003.3081_Optimal_hierarchical_modular_topologies_for_producing_limited_sustained_activati.pdf" 
    test_single_page_extraction(test_pdf_path, page_number=3)