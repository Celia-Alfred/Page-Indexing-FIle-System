
from pathlib import Path

import pymupdf

from services.pdf_service import extract_pages
from services.citation_service import find_ocr_citation_boxes


PDF_PATH = "sample_mixed.pdf"
OUTPUT_PATH = "multiline_citation_test.pdf"


def test_multiline_citation():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(PDF_PATH)

    pdf_bytes = pdf_path.read_bytes()
    extracted_pages = extract_pages(pdf_bytes)

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    print("\n========== MULTI-LINE CITATION TEST ==========")

    test_completed = False

    for page_data in extracted_pages:

        if page_data["extraction_source"] != "ocr":
            continue

        lines = page_data.get("lines", [])

        if len(lines) < 2:
            continue

        citation_text = (
            lines[0]["text"] + " " + lines[1]["text"]
        )

        print("\nPage:", page_data["page_number"])
        print("Test citation:", citation_text)

        result = find_ocr_citation_boxes(
            lines,
            citation_text
        )

        print("Match found:", result["found"])
        print("Matched lines:", len(result["matched_lines"]))
        print("Bounding boxes:", result["bounding_boxes"])

        if not result["found"]:
            continue

        page = document[page_data["page_number"] - 1]

        for bbox in result["bounding_boxes"]:
            page.add_highlight_annot(
                pymupdf.Rect(bbox)
            )

        test_completed = True
        break

    if test_completed:
        document.save(OUTPUT_PATH)
        print("\nOutput saved:", OUTPUT_PATH)
    else:
        print("\nNo suitable multi-line OCR match found.")

    document.close()


if __name__ == "__main__":
    test_multiline_citation()
