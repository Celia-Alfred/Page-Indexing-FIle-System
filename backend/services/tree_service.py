
import json
import re
from statistics import median
from typing import Any

from services.llm_service import generate_content_with_retry


MAX_BLOCK_CHARS = 700
MAX_OUTLINE_CHARS = 28_000
MAX_HEADING_WORDS = 24
MIN_FONT_RATIO = 1.15


def _clean_text(value: Any) -> str:
    """Normalize whitespace without changing the meaning of the text."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _get_block_text(block: dict) -> str:
    """Read text from the block structure produced by pdf_service.py."""
    text = _clean_text(block.get("text", ""))
    if text:
        return text

    parts = []
    for line in block.get("lines", []):
        if isinstance(line, dict):
            line_text = _clean_text(line.get("text", ""))
            if line_text:
                parts.append(line_text)

    return " ".join(parts)


def _get_font_sizes(block: dict) -> list[float]:
    """Support current and older span metadata field names."""
    sizes = []

    for line in block.get("lines", []):
        if not isinstance(line, dict):
            continue

        for span in line.get("spans", []):
            if not isinstance(span, dict):
                continue

            size = span.get("font_size", span.get("size", 0))

            try:
                size = float(size)
            except (TypeError, ValueError):
                continue

            if size > 0:
                sizes.append(size)

    return sizes


def _is_bold(block: dict) -> bool:
    """Inspect font names and PyMuPDF bold flags."""
    for line in block.get("lines", []):
        if not isinstance(line, dict):
            continue

        for span in line.get("spans", []):
            if not isinstance(span, dict):
                continue

            font_name = str(span.get("font", "")).lower()

            try:
                flags = int(span.get("flags", 0))
            except (TypeError, ValueError):
                flags = 0

            # PyMuPDF uses bit 4 for bold.
            if "bold" in font_name or flags & 16:
                return True

    return False


def _prepare_document(
    pages: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Flatten page blocks while preserving page and block indexes.
    These positions allow the tree to assign text to sections precisely.
    """
    prepared_pages = []
    records = []

    for page_position, page in enumerate(pages):
        try:
            page_number = int(
                page.get("page_number", page_position + 1)
            )
        except (TypeError, ValueError):
            page_number = page_position + 1

        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []

        page_records = []

        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            text = _get_block_text(block)
            if not text:
                continue

            record = {
                "page_number": page_number,
                "block_index": block_index,
                "text": text,
                "font_sizes": _get_font_sizes(block),
                "bold": _is_bold(block),
            }

            page_records.append(record)
            records.append(record)

        # Preserve text when block-level metadata is unavailable.
        if not page_records:
            page_text = _clean_text(page.get("text", ""))

            if page_text:
                record = {
                    "page_number": page_number,
                    "block_index": 0,
                    "text": page_text,
                    "font_sizes": [],
                    "bold": False,
                }
                page_records.append(record)
                records.append(record)

        prepared_pages.append({
            "page_number": page_number,
            "records": page_records,
        })

    return prepared_pages, records


def _looks_like_heading(text: str) -> bool:
    """Conservative local heading detector used as a fallback."""
    text = _clean_text(text)

    if not text or len(text) > 180:
        return False

    words = text.split()
    if not words or len(words) > MAX_HEADING_WORDS:
        return False

    if text.endswith((".", ",", ";", "?", "!")):
        return False

    if "://" in text or "www." in text.lower():
        return False

    if re.fullmatch(r"[\d\s.,:%/()\-]+", text):
        return False

    numbered = bool(
        re.match(
            r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|"
            r"[IVXLCDM]+[.)])\s+\S",
            text,
        )
    )

    uppercase = (
        any(character.isalpha() for character in text)
        and text == text.upper()
    )

    return numbered or uppercase


def _fallback_outline(records: list[dict]) -> list[dict]:
    """
    Infer a conservative outline from numbering and formatting when
    Gemini outline inference is unavailable.
    """
    sizes = [
        size
        for record in records
        for size in record["font_sizes"]
        if size > 0
    ]
    typical_size = median(sizes) if sizes else 0

    outline = []

    for record in records:
        text = record["text"]

        numbered = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|"
                r"[IVXLCDM]+[.)])\s+\S",
                text,
            )
        )

        larger_font = (
            typical_size > 0
            and record["font_sizes"]
            and max(record["font_sizes"])
            >= typical_size * MIN_FONT_RATIO
        )

        short_bold = (
            record["bold"]
            and len(text.split()) <= 12
            and not text.endswith((".", ",", ";", "?", "!"))
        )

        if not (
            _looks_like_heading(text)
            or (
                larger_font
                and len(text.split()) <= MAX_HEADING_WORDS
            )
            or short_bold
        ):
            continue

        level = 1

        decimal_match = re.match(r"^(\d+(?:\.\d+)+)", text)

        if decimal_match:
            level = min(
                decimal_match.group(1).count(".") + 1,
                6,
            )
        elif re.match(r"^\d+[.)]\s+", text):
            level = 1
        elif re.match(r"^[A-Z][.)]\s+", text):
            level = 2
        elif re.match(r"^[IVXLCDM]+[.)]\s+", text):
            level = 1

        outline.append({
            "title": text,
            "level": level,
            "start_page": record["page_number"],
            "start_block": record["block_index"],
        })

    return _deduplicate_outline(outline)


def _deduplicate_outline(outline: list[dict]) -> list[dict]:
    """Remove duplicate headings at the same position and adjacent repeats."""
    result = []
    seen_at_position = set()

    for item in outline:
        title = _clean_text(item.get("title", ""))
        if not title:
            continue

        try:
            page = int(item.get("start_page", 1))
            block = int(item.get("start_block", 0))
            level = max(1, min(int(item.get("level", 1)), 6))
        except (TypeError, ValueError):
            continue

        key = (page, block, title.casefold())
        if key in seen_at_position:
            continue

        seen_at_position.add(key)

        result.append({
            "title": title,
            "level": level,
            "start_page": page,
            "start_block": block,
        })

    result.sort(
        key=lambda item: (
            item["start_page"],
            item["start_block"],
        )
    )

    deduplicated = []

    for item in result:
        if (
            deduplicated
            and deduplicated[-1]["title"].casefold()
            == item["title"].casefold()
        ):
            continue

        deduplicated.append(item)

    return deduplicated


def _create_outline_prompt(records: list[dict]) -> str:
    """Prepare document evidence for Gemini without fixed heading templates."""
    lines = []

    for record in records:
        text = record["text"][:MAX_BLOCK_CHARS]
        metadata = []

        if record["font_sizes"]:
            metadata.append(
                "font_sizes="
                + ",".join(
                    f"{size:.1f}"
                    for size in record["font_sizes"][:4]
                )
            )

        if record["bold"]:
            metadata.append("bold=true")

        metadata_text = (
            f" [{', '.join(metadata)}]" if metadata else ""
        )

        lines.append(
            f'PAGE {record["page_number"]}, '
            f'BLOCK {record["block_index"]}'
            f'{metadata_text}: {text}'
        )

    evidence = "\n".join(lines)

    if len(evidence) > MAX_OUTLINE_CHARS:
        evidence = (
            evidence[:MAX_OUTLINE_CHARS]
            + "\n[Further document content omitted from this request.]"
        )

    return f"""
You are identifying the actual hierarchical structure of a PDF document.

Infer the structure from the supplied evidence. Do not assume that the
document is a research paper, book, report, manual, or any specific type.

Identify only headings supported by the text or formatting evidence.
Preserve the document's own heading wording. Do not invent section names.

Rules:
- Identify genuine titles, chapters, sections, and subsections.
- Level 1 is a top-level section; level 2 is its child, and so on.
- Consider numbering, typography, capitalization, wording, and context.
- A table or figure caption, running header, footer, author name,
  reference entry, or ordinary sentence is not automatically a heading.
- Do not create a heading just because a page starts with a sentence.
- Do not assume every page is a separate section.
- If no reliable headings exist, return an empty headings list.
- Every heading must refer to an exact PAGE and BLOCK in the evidence.
- Return headings in reading order.
- Do not generate summaries or page ranges; these are handled separately.
- Return valid JSON only.

Required JSON schema:
{{
  "document_title": "title supported by the document, or empty string",
  "headings": [
    {{
      "title": "exact heading text",
      "level": 1,
      "start_page": 1,
      "start_block": 0
    }}
  ]
}}

DOCUMENT EVIDENCE:
{evidence}
"""


def _infer_outline(
    records: list[dict],
) -> tuple[str, list[dict]]:
    """Ask Gemini to infer and validate the document outline."""
    if not records:
        return "", []

    response = generate_content_with_retry(
        _create_outline_prompt(records),
        config={"response_mime_type": "application/json"},
    )

    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty document outline")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid outline JSON"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Gemini returned an unexpected outline format"
        )

    document_title = _clean_text(
        result.get("document_title", "")
    )
    headings = result.get("headings", [])

    if not isinstance(headings, list):
        raise RuntimeError("Gemini returned an invalid headings list")

    valid_positions = {
        (record["page_number"], record["block_index"])
        for record in records
    }

    validated = []

    for heading in headings:
        if not isinstance(heading, dict):
            continue

        title = _clean_text(heading.get("title", ""))

        try:
            level = int(heading.get("level", 1))
            page = int(heading.get("start_page"))
            block = int(heading.get("start_block"))
        except (TypeError, ValueError):
            continue

        if (
            not title
            or len(title.split()) > MAX_HEADING_WORDS
            or not 1 <= level <= 6
            or (page, block) not in valid_positions
        ):
            continue

        validated.append({
            "title": title,
            "level": level,
            "start_page": page,
            "start_block": block,
        })

    return document_title, _deduplicate_outline(validated)


def _make_page_node(
    page_number: int,
    blocks: list[dict],
    document_id: str,
) -> dict:
    """Create a page node containing only its assigned text blocks."""
    content = "\n".join(
        block["text"]
        for block in blocks
        if block["text"]
    ).strip()

    return {
        "node_id": f"{document_id}_page_{page_number}",
        "node_type": "page",
        "title": f"Page {page_number}",
        "page_number": page_number,
        "start_page": page_number,
        "end_page": page_number,
        "content": content,
        "summary": "",
        "children": [],
    }


def _build_hierarchy(
    document_id: str,
    filename: str,
    document_title: str,
    pages: list[dict],
    records: list[dict],
    outline: list[dict],
) -> dict:
    """
    Build the document tree without generating section summaries.

    Text blocks are assigned to the deepest matching section. Page nodes
    contain only the text assigned to that section, preventing full-page
    duplication across unrelated sections.
    """
    page_numbers = [
        page["page_number"]
        for page in pages
    ]

    if not page_numbers:
        return {
            "node_id": document_id,
            "node_type": "document",
            "title": document_title or filename,
            "filename": filename,
            "start_page": 0,
            "end_page": 0,
            "summary": "",
            "children": [],
        }

    # If no reliable headings exist, preserve the document in one section.
    if not outline:
        outline = [{
            "title": "Document content",
            "level": 1,
            "start_page": page_numbers[0],
            "start_block": -1,
        }]

    sections = []

    for index, heading in enumerate(outline):
        sections.append({
            "node_id": f"{document_id}_section_{index + 1}",
            "node_type": "section",
            "title": heading["title"],
            "level": heading["level"],
            "start_page": heading["start_page"],
            "start_block": heading["start_block"],
            "end_page": page_numbers[-1],
            "summary": "",
            "content": "",
            "children": [],
        })

    # Find the end position for each section. A heading at the same
    # or a higher level closes the current section.
    section_end_positions = {}

    for index, section in enumerate(sections):
        end_position = None

        for next_section in sections[index + 1:]:
            if next_section["level"] <= section["level"]:
                end_position = (
                    next_section["start_page"],
                    next_section["start_block"],
                )
                break

        section_end_positions[section["node_id"]] = end_position

        if end_position is not None:
            section["end_page"] = end_position[0]
        else:
            section["end_page"] = page_numbers[-1]

    # Create nested section hierarchy.
    roots = []
    stack = []

    for section in sections:
        while stack and stack[-1]["level"] >= section["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(section)
        else:
            roots.append(section)

        stack.append(section)

    # Assign each text block to the deepest applicable section.
    assigned = {
        section["node_id"]: []
        for section in sections
    }

    for record in records:
        position = (
            record["page_number"],
            record["block_index"],
        )

        candidates = []

        for section in sections:
            start_position = (
                section["start_page"],
                section["start_block"],
            )
            end_position = section_end_positions[
                section["node_id"]
            ]

            if position < start_position:
                continue

            if (
                end_position is not None
                and position >= end_position
            ):
                continue

            candidates.append(section)

        if candidates:
            owner = max(
                candidates,
                key=lambda item: (
                    item["level"],
                    item["start_page"],
                    item["start_block"],
                ),
            )
            assigned[owner["node_id"]].append(record)

        else:
            # Preserve introductory text before the first detected heading.
            first_section = sections[0]
            assigned[first_section["node_id"]].append(record)

    # Add page nodes with their assigned text.
    for section in sections:
        section_records = assigned[section["node_id"]]

        section["content"] = "\n".join(
            record["text"]
            for record in section_records
        ).strip()

        by_page = {}

        for record in section_records:
            by_page.setdefault(
                record["page_number"], []
            ).append(record)

        for page_number, page_records in sorted(by_page.items()):
            section["children"].append(
                _make_page_node(
                    page_number,
                    page_records,
                    section["node_id"],
                )
            )

    # No Gemini summary calls are made during tree construction.
    return {
        "node_id": document_id,
        "node_type": "document",
        "title": document_title or filename,
        "filename": filename,
        "start_page": page_numbers[0],
        "end_page": page_numbers[-1],
        "summary": "",
        "children": roots,
    }


def build_page_index_tree(
    document_id: str,
    filename: str,
    pages: list[dict],
) -> dict:
    """
    Build a document-adaptive hierarchical index.

    Gemini infers headings and levels. If outline inference fails, a
    conservative local detector is used. No section summaries are generated.
    """
    prepared_pages, records = _prepare_document(pages)

    if not records:
        return {
            "node_id": document_id,
            "node_type": "document",
            "title": filename,
            "filename": filename,
            "start_page": 0,
            "end_page": 0,
            "summary": "",
            "children": [],
            "warning": "No extractable text was found in this PDF.",
        }

    document_title = ""
    outline = []

    try:
        document_title, outline = _infer_outline(records)
    except Exception as exc:
        print(f"Document outline inference failed: {exc}")
        outline = _fallback_outline(records)

    return _build_hierarchy(
        document_id=document_id,
        filename=filename,
        document_title=document_title,
        pages=prepared_pages,
        records=records,
        outline=outline,
    )
