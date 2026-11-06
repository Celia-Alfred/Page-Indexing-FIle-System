
from pathlib import Path
from typing import Any
from paddleocr import PaddleOCR

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    return _ocr_engine


def _get_result_data(result: Any) -> dict:
    """Extract structured OCR output from a PaddleOCR result."""

    output = result.json

    if callable(output):
        output = output()

    return output.get("res", output)


def extract_text_from_image(image_path: str) -> dict:
    """
    Extract text and bounding boxes from an image.

    Bounding boxes use image-pixel coordinates:
    [x1, y1, x2, y2]
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"OCR image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError("The supplied path must be an image file.")

    engine = get_ocr_engine()

    results = engine.predict(input=str(image_path))

    if not results:
        return {
            "text": "",
            "lines": [],
            "line_count": 0,
            "coordinate_system": "image_pixels",
        }

    data = _get_result_data(results[0])

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])

    lines = []

    for index, detected_text in enumerate(texts):

        text = str(detected_text).strip()

        if not text:
            continue

        confidence = None

        if index < len(scores):
            confidence = float(scores[index])

        bbox = None

        if index < len(boxes):
            coordinates = boxes[index]

            if len(coordinates) == 4:
                bbox = [
                    float(coordinate)
                    for coordinate in coordinates
                ]

        lines.append({
            "line_no": len(lines),
            "block_no": len(lines),
            "text": text,
            "confidence": confidence,
            "bbox": bbox,
        })

    extracted_text = "\n".join(
        line["text"] for line in lines
    )

    return {
        "text": extracted_text,
        "lines": lines,
        "line_count": len(lines),
        "coordinate_system": "image_pixels",
    }
