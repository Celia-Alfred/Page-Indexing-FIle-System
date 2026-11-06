
import pymupdf

from pathlib import Path
from services.ocr_service import extract_text_from_image


PDF_PATH = "sample_scanned.pdf"
IMAGE_PATH = "rendered_page.png"

# Select the first page (zero-based index)
PAGE_INDEX = 0

# Render resolution
RENDER_SCALE = 2


def test_pdf_ocr():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Open PDF
    document = pymupdf.open(PDF_PATH)

    if PAGE_INDEX >= len(document):
        raise ValueError("Selected page does not exist.")

    page = document[PAGE_INDEX]

    # Get original page dimensions
    page_width = page.rect.width
    page_height = page.rect.height

    # Render PDF page as an image
    matrix = pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    pixmap.save(IMAGE_PATH)

    print("\nPDF PAGE DETAILS")
    print("----------------")
    print("Page number:", PAGE_INDEX + 1)
    print("Page width:", page_width)
    print("Page height:", page_height)
    print("Rendered image width:", pixmap.width)
    print("Rendered image height:", pixmap.height)

    # Run OCR on rendered image
    result = extract_text_from_image(IMAGE_PATH)

    print("\n========== PDF OCR RESULT ==========\n")

    print("Extracted text:\n")
    print(result["text"])

    print("\nTotal detected lines:", result["line_count"])

    print("\nBounding boxes:")
    for line in result["lines"]:
        print(line)

    print("\nCoordinate system:", result["coordinate_system"])

    print("\n====================================")

    document.close()


if __name__ == "__main__":
    test_pdf_ocr()
