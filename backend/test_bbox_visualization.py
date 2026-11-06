
from pathlib import Path

import pymupdf

from services.pdf_service import extract_pages


PDF_PATH = "sample_scanned.pdf"
OUTPUT_PATH = "bbox_test_output.pdf"


def test_bbox_visualization():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf_bytes = pdf_path.read_bytes()

    # Run the actual hybrid extraction service
    extracted_pages = extract_pages(pdf_bytes)

    # Open a separate copy for annotation
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    print("\n========== BBOX VISUALIZATION TEST ==========")

    for page_data in extracted_pages:

        page_number = page_data["page_number"]
        page = document[page_number - 1]

        if page_data["extraction_source"] != "ocr":
            print(f"Page {page_number}: Native text - skipped")
            continue

        boxes_drawn = 0

        for line in page_data["lines"]:

            bbox = line.get("bbox")

            if not bbox:
                continue

            rect = pymupdf.Rect(bbox)

            if rect.is_empty or not rect.is_valid:
                continue

            page.draw_rect(
                rect,
                color=(1, 0, 0),
                width=1.0,
                overlay=True
            )

            boxes_drawn += 1

        print(
            f"Page {page_number}: "
            f"{boxes_drawn} OCR bounding boxes drawn"
        )

    document.save(OUTPUT_PATH)
    document.close()

    print("\nOutput saved:", OUTPUT_PATH)


if __name__ == "__main__":
    test_bbox_visualization()
