from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pymupdf

from services.ocr_service import extract_text_from_image


MIN_NATIVE_TEXT_LENGTH = 30

RENDER_SCALE = 2


def _map_ocr_bbox_to_pdf(
    bbox: list,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> list | None:
    """
    Convert OCR image-pixel coordinates into PDF page-point coordinates.

    This mapping is valid for the unrotated page coordinate frame
    used by the rendered image. Rotation is handled separately.
    """

    if not bbox or len(bbox) != 4:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    x_scale = page_width / image_width
    y_scale = page_height / image_height

    x1, y1, x2, y2 = bbox

    return [
        float(x1 * x_scale),
        float(y1 * y_scale),
        float(x2 * x_scale),
        float(y2 * y_scale),
    ]


def _extract_native_page(page: Any) -> dict:
    """
    Extract native text, blocks, lines, and word coordinates.
    Coordinates are in PDF page-point coordinates.
    """

    page_data = page.get_text("dict")

    blocks = []
    lines = []
    words = []

    for block_index, block in enumerate(page_data.get("blocks", [])):

        if block.get("type") != 0:
            continue

        block_lines = []

        for line in block.get("lines", []):

            spans = line.get("spans", [])

            line_text = "".join(
                span.get("text", "") for span in spans
            ).strip()

            if not line_text:
                continue

            bbox = [
                float(value)
                for value in line.get("bbox", [])
            ]

            block_lines.append(line_text)

            lines.append({
                "line_no": len(lines),
                "block_no": block_index,
                "text": line_text,
                "bbox": bbox,
                "coordinate_system": "pdf_points",
            })

        block_text = "\n".join(block_lines)

        if block_text:
            blocks.append({
                "block_no": block_index,
                "text": block_text,
                "bbox": [
                    float(value)
                    for value in block.get("bbox", [])
                ],
                "coordinate_system": "pdf_points",
            })

    for word_data in page.get_text("words"):

        x0, y0, x1, y1 = word_data[:4]

        words.append({
            "text": word_data[4],
            "bbox": [
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            ],
            "block_no": int(word_data[5]),
            "line_no": int(word_data[6]),
            "word_no": int(word_data[7]),
            "coordinate_system": "pdf_points",
        })

    return {
        "text": page.get_text("text").strip(),
        "blocks": blocks,
        "lines": lines,
        "words": words,
        "line_count": len(lines),
        "word_count": len(words),
        "coordinate_system": "pdf_points",
    }


def _extract_ocr_page(page: Any) -> dict:
    """
    Render a scanned PDF page and extract text using PaddleOCR.

    Original OCR boxes are retained in image-pixel coordinates.
    Additional mapped boxes are provided in PDF page-point coordinates.
    """

    matrix = pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_path = Path(temp_file.name)

    try:
        pixmap.save(str(image_path))

        ocr_result = extract_text_from_image(str(image_path))

    finally:
        image_path.unlink(missing_ok=True)

    image_width = pixmap.width
    image_height = pixmap.height

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    ocr_lines = []

    for line in ocr_result.get("lines", []):

        original_bbox = line.get("bbox")

        pdf_bbox = _map_ocr_bbox_to_pdf(
            bbox=original_bbox,
            page_width=page_width,
            page_height=page_height,
            image_width=image_width,
            image_height=image_height,
        )

        ocr_lines.append({
            **line,

            "image_bbox": original_bbox,

            "pdf_bbox": pdf_bbox,

            "coordinate_system": "image_pixels",
            "pdf_coordinate_system": "pdf_points",
        })

    blocks = []

    for line in ocr_lines:
        blocks.append({
            "block_no": line["block_no"],
            "text": line["text"],
            "bbox": line["pdf_bbox"],
            "image_bbox": line["image_bbox"],
            "confidence": line["confidence"],
            "coordinate_system": "pdf_points",
        })

    return {
        "text": ocr_result["text"],
        "blocks": blocks,
        "lines": ocr_lines,
        "words": [],
        "line_count": len(ocr_lines),
        "word_count": 0,
        "coordinate_system": "pdf_points",
        "original_coordinate_system": "image_pixels",
        "image_width": image_width,
        "image_height": image_height,
        "render_scale": RENDER_SCALE,
    }


def extract_pdf_hybrid(
    pdf_path: str,
    min_native_text_length: int = MIN_NATIVE_TEXT_LENGTH,
) -> dict:
    """
    Extract a PDF using page-level hybrid extraction.

    Digital pages:
        PyMuPDF.

    Scanned pages:
        PaddleOCR.

    Returns document metadata and page-level extraction results.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            "The supplied path must be a PDF file."
        )

    if min_native_text_length < 1:
        raise ValueError(
            "min_native_text_length must be at least 1."
        )

    document = pymupdf.open(str(pdf_path))

    pages = []

    try:
        for page_index, page in enumerate(document):

            page_number = page_index + 1

            native_text = page.get_text("text").strip()

            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            page_rotation = int(page.rotation)

            if len(native_text) >= min_native_text_length:

                result = _extract_native_page(page)
                extraction_source = "native"

            else:

                result = _extract_ocr_page(page)
                extraction_source = "ocr"

            page_result = {
                "page_number": page_number,
                "extraction_source": extraction_source,
                "text": result["text"],
                "blocks": result["blocks"],
                "lines": result["lines"],
                "words": result["words"],
                "line_count": result["line_count"],
                "word_count": result["word_count"],
                "page_width": page_width,
                "page_height": page_height,
                "page_rotation": page_rotation,
                "coordinate_system": result["coordinate_system"],
                "original_coordinate_system": result.get(
                    "original_coordinate_system"
                ),
                "image_width": result.get("image_width"),
                "image_height": result.get("image_height"),
                "render_scale": result.get("render_scale"),
            }

            pages.append(page_result)

    finally:
        document.close()

    return {
        "filename": pdf_path.name,
        "total_pages": len(pages),
        "pages": pages,
    }