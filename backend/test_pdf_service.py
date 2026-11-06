
from pathlib import Path

from services.pdf_service import extract_pages


PDF_PATH = "sample_scanned.pdf"


def test_pdf_service():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    pdf_bytes = pdf_path.read_bytes()

    pages = extract_pages(pdf_bytes)

    print("\n========== PDF SERVICE TEST ==========")
    print("Total pages:", len(pages))

    for page in pages:

        print(f"\n========== PAGE {page['page_number']} ==========")

        print("Source:", page["extraction_source"])
        print("Text:", page["text"][:400])

        print("Blocks:", len(page["blocks"]))
        print("Lines:", page["line_count"])
        print("Words:", page["word_count"])

        print("Page dimensions:",
              page["page_width"],
              "x",
              page["page_height"])

        print("Rotation:", page["page_rotation"])
        print("Coordinate system:", page["coordinate_system"])

        if page["extraction_source"] == "ocr":

            if page["lines"]:

                print("\nFirst OCR line:")
                print(page["lines"][0])

                print("\nOriginal image coordinates:")
                print(page["lines"][0]["image_bbox"])

                print("\nMapped PDF coordinates:")
                print(page["lines"][0]["bbox"])

        elif page["words"]:

            print("\nFirst native word:")
            print(page["words"][0])

    print("\n========== SUMMARY ==========")

    for page in pages:

        print(
            f"Page {page['page_number']}: "
            f"{page['extraction_source'].upper()} | "
            f"{page['line_count']} lines | "
            f"{page['word_count']} words"
        )


if __name__ == "__main__":
    test_pdf_service()
