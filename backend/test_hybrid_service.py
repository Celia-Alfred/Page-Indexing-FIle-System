
from pathlib import Path

from services.hybrid_extraction_service import extract_pdf_hybrid


PDF_PATH = "sample_mixed.pdf"


def test_hybrid_service():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    result = extract_pdf_hybrid(str(pdf_path))

    print("\n========== HYBRID SERVICE TEST ==========")
    print("Filename:", result["filename"])
    print("Total pages:", result["total_pages"])

    for page in result["pages"]:

        print(f"\n========== PAGE {page['page_number']} ==========")

        print("Extraction source:", page["extraction_source"])

        print("\nExtracted text:")
        print(page["text"][:500])

        print("\nLine count:", page["line_count"])
        print("Word count:", page["word_count"])

        print(
            "\nPage dimensions:",
            f"{page['page_width']} x {page['page_height']}"
        )

        print("Rotation:", page["page_rotation"])
        print("Coordinate system:", page["coordinate_system"])

        # Inspect OCR coordinates.
        if page["extraction_source"] == "ocr":

            print("\nImage dimensions:")
            print(
                page["image_width"],
                "x",
                page["image_height"]
            )

            print("\nRender scale:")
            print(page["render_scale"])

            if page["lines"]:

                print("\nFirst OCR line:")
                print(page["lines"][0])

                print("\nOriginal image coordinates:")
                print(page["lines"][0]["image_bbox"])

                print("\nMapped PDF coordinates:")
                print(page["lines"][0]["pdf_bbox"])

            else:
                print("\nNo OCR lines detected.")

        # Inspect native word coordinates.
        elif page["extraction_source"] == "native":

            if page["words"]:

                print("\nFirst native word:")
                print(page["words"][0])

                print("\nNative word coordinates:")
                print(page["words"][0]["bbox"])

            else:
                print("\nNo native words detected.")

        print("\nBlocks:", len(page["blocks"]))
        print("Lines:", len(page["lines"]))
        print("Words:", len(page["words"]))

    print("\n========== EXTRACTION SUMMARY ==========")

    for page in result["pages"]:

        print(
            f"Page {page['page_number']}: "
            f"{page['extraction_source'].upper()} | "
            f"{page['line_count']} lines | "
            f"{page['word_count']} words"
        )

    print("\n========================================")


if __name__ == "__main__":
    test_hybrid_service()
