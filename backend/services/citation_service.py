
import re
import unicodedata

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

def _normalize_tokens(text: str) -> list[str]:
    """Normalize text and return comparable word tokens."""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).casefold()

    return WORD_PATTERN.findall(normalized)

def find_ocr_citation_boxes(
    lines: list[dict],
    citation_text: str,
) -> dict:
    """
    Find an exact quotation in OCR-extracted lines.

    The quotation may span multiple consecutive OCR lines.

    Returns the matching lines and their PDF-coordinate
    bounding boxes.
    """

    if not isinstance(lines, list):
        return {
            "found": False,
            "matched_lines": [],
            "bounding_boxes": [],
        }

    citation_tokens = _normalize_tokens(citation_text)

    if not citation_tokens:
        return {
            "found": False,
            "matched_lines": [],
            "bounding_boxes": [],
        }

    document_tokens = []
    for line_index, line in enumerate(lines):

        if not isinstance(line, dict):
            continue
        line_tokens = _normalize_tokens(
            line.get("text", "")
        )

        for token in line_tokens:
            document_tokens.append({
                "token": token,
                "line_index": line_index,
            })

    target_length = len(citation_tokens)

    if len(document_tokens) < target_length:
        return {
            "found": False,
            "matched_lines": [],
            "bounding_boxes": [],
        }

    for start in range(
        len(document_tokens) - target_length + 1
    ):

        matched_tokens = document_tokens[
            start:start + target_length
        ]

        candidate = [
            item["token"]
            for item in matched_tokens
        ]

        if candidate != citation_tokens:
            continue

        matched_indexes = sorted({
            item["line_index"]
            for item in matched_tokens
        })

        matched_lines = [
            lines[index]
            for index in matched_indexes
        ]

        bounding_boxes = [
            line["bbox"]
            for line in matched_lines
            if line.get("bbox") is not None
        ]

        return {
            "found": True,
            "matched_lines": matched_lines,
            "bounding_boxes": bounding_boxes,
        }

    return {
        "found": False,
        "matched_lines": [],
        "bounding_boxes": [],
    }
