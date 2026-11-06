
import pymupdf
from pathlib import Path

from services.ocr_service import extract_text_from_image
PDF_PATH = "sample_mixed.pdf"
RENDER_SCALE = 2

MIN_NATIVE_TEXT_LENGTH = 30


def extract_page_text(page, page_number):
    """
    Select native PDF extraction or OCR independently for each page.
    """

    native_text = page.get_text("text").strip()

    print(f"\n========== PAGE {page_number} ==========")

    if len(native_text) >= MIN_NATIVE_TEXT_LENGTH:
        print("Extraction method: Native PyMuPDF")
        return {
            "page_number": page_number,
            "source": "native",
            "text": native_text,
            "line_count": len(native_text.splitlines()),
            "coordinate_system": "not_mapped",
        }

    print("Extraction method: PaddleOCR")

    matrix = pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    image_path = f"temp_page_{page_number}.png"
    pixmap.save(image_path)

    try:
        ocr_result = extract_text_from_image(image_path)

        return {
            "page_number": page_number,
            "source": "ocr",
            "text": ocr_result["text"],
            "line_count": ocr_result["line_count"],
            "lines": ocr_result["lines"],
            "coordinate_system": ocr_result["coordinate_system"],
            "image_width": pixmap.width,
            "image_height": pixmap.height,
            "render_scale": RENDER_SCALE,
        }

    finally:
        Path(image_path).unlink(missing_ok=True)


def test_hybrid_extraction():

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    document = pymupdf.open(PDF_PATH)

    print("\n========== HYBRID EXTRACTION TEST ==========")
    print("Total PDF pages:", len(document))

    results = []

    for page_index, page in enumerate(document):
        result = extract_page_text(page, page_index + 1)
        results.append(result)

        print("\nExtracted text:")
        print(result["text"])
        print("\nDetected lines:", result["line_count"])
        print("Coordinate system:", result["coordinate_system"])

    print("\n========== EXTRACTION SUMMARY ==========")

    for result in results:
        print(
            f"Page {result['page_number']}: "
            f"{result['source'].upper()} | "
            f"{result['line_count']} lines"
        )

    document.close()


if __name__ == "__main__":
    test_hybrid_extraction()
