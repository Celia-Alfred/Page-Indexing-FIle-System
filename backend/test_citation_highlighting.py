
from pathlib import Path

import pymupdf

from services.pdf_service import extract_pages


PDF_PATH = "sample_mixed.pdf"
OUTPUT_PATH = "citation_highlight_test.pdf"


def test_citation_highlighting():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf_bytes = pdf_path.read_bytes()

    # Extract using our actual hybrid extraction service
    extracted_pages = extract_pages(pdf_bytes)

    # Open original PDF for annotations
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    print("\n========== CITATION HIGHLIGHTING TEST ==========")

    highlighted_count = 0

    for page_data in extracted_pages:

        page_number = page_data["page_number"]
        page = document[page_number - 1]

        source = page_data["extraction_source"]

        if source == "ocr":

            lines = page_data.get("lines", [])

            if not lines:
                print(f"Page {page_number}: No OCR lines found")
                continue

            selected_line = lines[0]
            bbox = selected_line.get("bbox")

            if not bbox:
                print(f"Page {page_number}: OCR bbox unavailable")
                continue

            rect = pymupdf.Rect(bbox)

            if rect.is_empty or not rect.is_valid:
                print(f"Page {page_number}: Invalid OCR bbox")
                continue

            page.add_highlight_annot(rect)

            print(f"\nPage {page_number} | OCR")
            print("Selected text:", selected_line["text"])
            print("Bounding box:", bbox)

            highlighted_count += 1

        elif source == "native":

            words = page_data.get("words", [])

            if not words:
                print(f"Page {page_number}: No native words found")
                continue

            selected_word = words[0]
            bbox = selected_word.get("bbox")

            if not bbox:
                print(f"Page {page_number}: Native bbox unavailable")
                continue

            rect = pymupdf.Rect(bbox)

            if rect.is_empty or not rect.is_valid:
                print(f"Page {page_number}: Invalid native bbox")
                continue

            page.add_highlight_annot(rect)

            print(f"\nPage {page_number} | NATIVE")
            print("Selected word:", selected_word["text"])
            print("Bounding box:", bbox)

            highlighted_count += 1

    document.save(OUTPUT_PATH)
    document.close()

    print("\n========== TEST SUMMARY ==========")
    print("Highlights created:", highlighted_count)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    test_citation_highlighting()
