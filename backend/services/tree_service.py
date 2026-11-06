
import json
import re
import time
from collections import deque
from statistics import median
from typing import Any

from services.llm_service import generate_content_with_retry

MAX_BLOCK_CHARS = 700
MAX_OUTLINE_CHARS = 28_000
MAX_OUTLINE_BATCH_CHARS = 3500

MAX_HEADING_WORDS = 24
MAX_HEADING_CHARS = 180

MIN_FONT_RATIO = 1.15
MAX_HEADING_LEVEL = 6


SAFE_INPUT_TPM = 5500


_OUTLINE_REQUEST_HISTORY = deque()


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
    """Extract font sizes from span metadata."""
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
    """Check font names and PyMuPDF bold flags."""
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

            if "bold" in font_name or flags & 16:
                return True

    return False


def _get_bbox(block: dict) -> list[float] | None:
    """Return a valid block bounding box when available."""
    bbox = block.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    x0, y0, x1, y1 = values

    if x1 <= x0 or y1 <= y0:
        return None

    return values


def _position_key(record: dict) -> tuple[int, int]:
    """Return the stable logical position used by the tree."""
    return (
        int(record["page_number"]),
        int(record["block_index"]),
    )


def _outline_position(item: dict) -> tuple[int, int]:
    return (
        int(item["start_page"]),
        int(item["start_block"]),
    )



def _prepare_document(
    pages: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Flatten extracted page blocks without losing layout information.

    block_index:
        Position in the extracted page's blocks array.

    block_no:
        Original PyMuPDF block number, retained as metadata.

    bbox:
        Original PDF coordinates used for layout understanding.
    """
    prepared_pages = []
    records = []

    for page_position, page in enumerate(pages):
        if not isinstance(page, dict):
            continue

        try:
            page_number = int(
                page.get("page_number", page_position + 1)
            )
        except (TypeError, ValueError):
            page_number = page_position + 1

        page_width = page.get("page_width")
        page_height = page.get("page_height")
        page_rotation = page.get("page_rotation", 0)

        page_records = []
        blocks = page.get("blocks", [])

        if not isinstance(blocks, list):
            blocks = []

        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            text = _get_block_text(block)

            if not text:
                continue

            try:
                block_no = int(block.get("block_no", block_index))
            except (TypeError, ValueError):
                block_no = block_index

            record = {
                "page_number": page_number,
                "block_index": block_index,
                "block_no": block_no,
                "block_id": block.get(
                    "block_id",
                    f"page-{page_number}-block-{block_no}",
                ),
                "text": text,
                "bbox": _get_bbox(block),
                "font_sizes": _get_font_sizes(block),
                "bold": _is_bold(block),
                "page_width": page_width,
                "page_height": page_height,
                "page_rotation": page_rotation,
            }

            page_records.append(record)
            records.append(record)

        if not page_records:
            page_text = _clean_text(page.get("text", ""))

            if page_text:
                record = {
                    "page_number": page_number,
                    "block_index": 0,
                    "block_no": 0,
                    "block_id": f"page-{page_number}-block-0",
                    "text": page_text,
                    "bbox": None,
                    "font_sizes": [],
                    "bold": False,
                    "page_width": page_width,
                    "page_height": page_height,
                    "page_rotation": page_rotation,
                }

                page_records.append(record)
                records.append(record)

        prepared_pages.append({
            "page_number": page_number,
            "page_width": page_width,
            "page_height": page_height,
            "page_rotation": page_rotation,
            "records": page_records,
        })

    prepared_pages.sort(key=lambda item: item["page_number"])
    records.sort(key=_position_key)

    return prepared_pages, records



def _looks_like_heading(text: str) -> bool:
    """Conservative heading detector for fallback outline generation."""
    text = _clean_text(text)

    if not text or len(text) > MAX_HEADING_CHARS:
        return False

    words = text.split()

    if not words or len(words) > MAX_HEADING_WORDS:
        return False

    if text.endswith((".", ",", ";", "?", "!", ":")):
        return False

    lowered = text.casefold()

    if "://" in lowered or "www." in lowered:
        return False

    if re.fullmatch(r"[\d\s.,:%/()\-]+", text):
        return False

    numbered = bool(
        re.match(
            r"^(?:"
            r"\d+(?:\.\d+)*[.)]?"
            r"|[A-Z][.)]"
            r"|[IVXLCDM]+[.)]"
            r")\s+\S",
            text,
        )
    )

    uppercase = (
        any(character.isalpha() for character in text)
        and text == text.upper()
    )

    return numbered or uppercase


def _detect_heading_level(text: str) -> int:
    """Estimate hierarchy level from common numbering patterns."""
    text = _clean_text(text)

    decimal_match = re.match(
        r"^(\d+(?:\.\d+)+)",
        text,
    )

    if decimal_match:
        return min(
            decimal_match.group(1).count(".") + 1,
            MAX_HEADING_LEVEL,
        )

    if re.match(r"^\d+[.)]\s+", text):
        return 1

    if re.match(r"^[IVXLCDM]+[.)]\s+", text):
        return 1

    if re.match(r"^[A-Z][.)]\s+", text):
        return 2

    return 1


def _fallback_outline(records: list[dict]) -> list[dict]:
    """
    Build a conservative outline when AI inference is unavailable.

    Uses numbering, font size, and bold formatting.
    """
    sizes = [
        size
        for record in records
        for size in record.get("font_sizes", [])
        if size > 0
    ]

    typical_size = median(sizes) if sizes else 0
    outline = []

    for record in records:
        text = record["text"]

        numbered = bool(
            re.match(
                r"^(?:"
                r"\d+(?:\.\d+)*[.)]?"
                r"|[A-Z][.)]"
                r"|[IVXLCDM]+[.)]"
                r")\s+\S",
                text,
            )
        )

        larger_font = (
            typical_size > 0
            and record.get("font_sizes")
            and max(record["font_sizes"])
            >= typical_size * MIN_FONT_RATIO
        )

        short_bold = (
            record.get("bold", False)
            and len(text.split()) <= 12
            and not text.endswith((".", ",", ";", "?", "!"))
        )

        if not (
            _looks_like_heading(text)
            or (
                larger_font
                and len(text.split()) <= MAX_HEADING_WORDS
                and len(text) <= MAX_HEADING_CHARS
            )
            or short_bold
        ):
            continue

        outline.append({
            "title": text,
            "level": _detect_heading_level(text),
            "start_page": record["page_number"],
            "start_block": record["block_index"],
        })

    return _deduplicate_outline(outline)


def _deduplicate_outline(outline: list[dict]) -> list[dict]:
    """Validate, sort, and remove duplicate outline entries."""
    result = []
    seen_positions = set()

    for item in outline:
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title", ""))

        try:
            page = int(item.get("start_page"))
            block = int(item.get("start_block"))
            level = int(item.get("level", 1))
        except (TypeError, ValueError):
            continue

        if (
            not title
            or len(title) > MAX_HEADING_CHARS
            or len(title.split()) > MAX_HEADING_WORDS
            or page < 1
            or block < 0
            or not 1 <= level <= MAX_HEADING_LEVEL
        ):
            continue

        key = (page, block, title.casefold())

        if key in seen_positions:
            continue

        seen_positions.add(key)

        result.append({
            "title": title,
            "level": level,
            "start_page": page,
            "start_block": block,
        })

    result.sort(key=_outline_position)

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
    """
    Give Groq text and layout evidence.

    Coordinates are supplied as evidence, not absolute rules.
    """
    lines = []

    for record in records:
        text = record["text"][:MAX_BLOCK_CHARS]

        metadata = []

        bbox = record.get("bbox")

        if bbox:
            x0, y0, x1, y1 = bbox
            metadata.append(
                f"bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
            )

        if record.get("page_width") and record.get("page_height"):
            metadata.append(
                "page_size="
                f"({record['page_width']},{record['page_height']})"
            )

        if record.get("page_rotation"):
            metadata.append(
                f"rotation={record['page_rotation']}"
            )

        if record.get("font_sizes"):
            metadata.append(
                "font_sizes="
                + ",".join(
                    f"{size:.1f}"
                    for size in record["font_sizes"][:4]
                )
            )

        if record.get("bold"):
            metadata.append("bold=true")

        metadata_text = (
            f" [{'; '.join(metadata)}]" if metadata else ""
        )

        lines.append(
            f"PAGE {record['page_number']}, "
            f"BLOCK {record['block_index']}"
            f"{metadata_text}: {text}"
        )

    evidence = "\n".join(lines)

    if len(evidence) > MAX_OUTLINE_CHARS:
        evidence = (
            evidence[:MAX_OUTLINE_CHARS]
            + "\n[Additional document content omitted.]"
        )

    return f"""
You are identifying the actual hierarchical structure of a PDF.

Infer the document structure from the supplied text and layout evidence.

Do not assume that the document is a research paper, book, report,
manual, presentation, or any other specific document type.

GENERAL RULES:

1. Identify genuine document titles, chapters, sections,
   subsections, and other meaningful structural headings.

2. Preserve the exact wording of headings.

3. Do not invent headings or section names.

4. Use levels consistently:
   level 1 = top-level heading,
   level 2 = subsection,
   level 3 = subsection of level 2, and so on.

5. Consider numbering, font size, bold formatting, capitalization,
   wording, surrounding text, and spatial position.

6. A paragraph, sentence, caption, table entry, reference entry,
   author name, running header, or footer is not automatically
   a heading.

7. Do not mistake a page boundary for a section boundary.

8. Documents may have multiple columns. Use bounding boxes to
   understand the layout. Blocks at similar vertical positions
   in different columns may belong to different sections.

9. Do not infer reading order from vertical position alone.

10. If a heading is part of a larger block, only return it as a
    heading if that block can be represented by an exact supplied
    block position. Do not invent a position.

11. Every heading must match an exact PAGE and BLOCK in the evidence.

12. Return headings in their best logical document order.

13. If the document has no reliable headings, return an empty list.

14. Return valid JSON only. Do not include Markdown fences.

Return this JSON structure:

{{
  "document_title": "document title supported by the evidence",
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


def _split_outline_batches(
    records: list[dict],
) -> list[list[dict]]:
    """Split records into smaller batches for Groq requests."""
    batches = []
    current_batch = []
    current_size = 0

    for record in records:
        record_size = len(record.get("text", "")) + 250

        if (
            current_batch
            and current_size + record_size
            > MAX_OUTLINE_BATCH_CHARS
        ):
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(record)
        current_size += record_size

    if current_batch:
        batches.append(current_batch)

    return batches


def _estimate_input_tokens(prompt: str) -> int:
    """
    Estimate input tokens conservatively.

    This is an approximation, not Groq's exact tokenizer.
    """
    return int(len(prompt) / 3 * 1.2) + 1


def _wait_for_groq_budget(prompt: str) -> None:
    """
    Pace outline requests against a conservative rolling
    60-second input-token budget.
    """
    estimated_tokens = _estimate_input_tokens(prompt)

    if estimated_tokens > SAFE_INPUT_TPM:
        raise ValueError(
            "Outline batch exceeds the configured Groq token budget."
        )

    while True:
        now = time.monotonic()

        while (
            _OUTLINE_REQUEST_HISTORY
            and now - _OUTLINE_REQUEST_HISTORY[0][0] >= 60
        ):
            _OUTLINE_REQUEST_HISTORY.popleft()

        used_tokens = sum(
            tokens
            for _, tokens in _OUTLINE_REQUEST_HISTORY
        )

        if used_tokens + estimated_tokens <= SAFE_INPUT_TPM:
            _OUTLINE_REQUEST_HISTORY.append(
                (now, estimated_tokens)
            )
            return

        oldest_request_time = _OUTLINE_REQUEST_HISTORY[0][0]

        wait_time = max(
            1,
            60 - (now - oldest_request_time) + 1,
        )

        print(
            f"Groq input token budget reached. "
            f"Waiting {wait_time:.0f} seconds..."
        )

        time.sleep(wait_time)


def _infer_outline(
    records: list[dict],
) -> tuple[str, list[dict]]:
    """
    Infer the document outline using multiple smaller requests.

    Every returned heading is validated against the original
    document records.
    """
    if not records:
        return "", []

    batches = _split_outline_batches(records)

    document_title = ""
    combined_outline = []

    valid_positions = {
        _position_key(record)
        for record in records
    }

    print(
        f"Generating document outline using "
        f"{len(batches)} batches..."
    )

    for batch_index, batch in enumerate(batches, start=1):
        print(
            f"Processing outline batch "
            f"{batch_index}/{len(batches)} "
            f"({len(batch)} blocks)..."
        )

        prompt = _create_outline_prompt(batch)

        _wait_for_groq_budget(prompt)

        response = generate_content_with_retry(
            prompt,
            config={"response_mime_type": "application/json"},
        )

        raw = (response.text or "").strip()

        if not raw:
            raise RuntimeError(
                "Groq returned an empty document outline"
            )

        try:
            result = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Groq returned invalid outline JSON"
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Groq returned an unexpected outline format"
            )

        if not document_title:
            document_title = _clean_text(
                result.get("document_title", "")
            )

        headings = result.get("headings", [])

        if not isinstance(headings, list):
            raise RuntimeError(
                "Groq returned an invalid headings list"
            )

        for heading in headings:
            if not isinstance(heading, dict):
                continue

            title = _clean_text(
                heading.get("title", "")
            )

            try:
                level = int(heading.get("level", 1))
                page = int(heading.get("start_page"))
                block = int(heading.get("start_block"))

            except (TypeError, ValueError):
                continue

            if (
                not title
                or len(title) > MAX_HEADING_CHARS
                or len(title.split()) > MAX_HEADING_WORDS
                or not 1 <= level <= MAX_HEADING_LEVEL
                or (page, block) not in valid_positions
            ):
                continue

            combined_outline.append({
                "title": title,
                "level": level,
                "start_page": page,
                "start_block": block,
            })

    return (
        document_title,
        _deduplicate_outline(combined_outline),
    )



def _make_page_node(
    page_number: int,
    blocks: list[dict],
    document_id: str,
) -> dict:
    """Create a page node containing its assigned text blocks."""
    content = "\n".join(
        block["text"]
        for block in blocks
        if block.get("text")
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


def _make_section_node(
    document_id: str,
    index: int,
    title: str,
    level: int,
    start_page: int,
    start_block: int,
) -> dict:
    return {
        "node_id": f"{document_id}_section_{index}",
        "node_type": "section",
        "title": title,
        "level": level,
        "start_page": start_page,
        "start_block": start_block,
        "end_page": start_page,
        "summary": "",
        "content": "",
        "children": [],
    }


def _attach_sections(
    outline: list[dict],
    document_id: str,
) -> tuple[list[dict], list[dict]]:
    """Convert a flat outline into a nested section hierarchy."""
    sections = []

    for index, heading in enumerate(outline, start=1):
        sections.append(
            _make_section_node(
                document_id=document_id,
                index=index,
                title=heading["title"],
                level=heading["level"],
                start_page=heading["start_page"],
                start_block=heading["start_block"],
            )
        )

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

    return sections, roots


def _calculate_section_end_positions(
    sections: list[dict],
    last_page: int,
) -> dict:
    """
    Calculate exclusive end positions.

    A section ends at the next heading of the same or higher level.
    A subsection does not end its parent section.
    """
    end_positions = {}

    for index, section in enumerate(sections):
        end_position = None

        for next_section in sections[index + 1:]:
            if next_section["level"] <= section["level"]:
                end_position = (
                    next_section["start_page"],
                    next_section["start_block"],
                )
                break

        end_positions[section["node_id"]] = end_position

        if end_position:
            section["end_page"] = end_position[0]
        else:
            section["end_page"] = last_page

    return end_positions


def _assign_records_to_sections(
    records: list[dict],
    sections: list[dict],
    end_positions: dict,
) -> tuple[dict, list[dict]]:
    """
    Assign each record to the deepest matching section.

    Content before the first heading is kept separately.
    """
    assigned = {
        section["node_id"]: []
        for section in sections
    }

    unclassified = []

    for record in records:
        position = _position_key(record)
        candidates = []

        for section in sections:
            start_position = (
                section["start_page"],
                section["start_block"],
            )

            end_position = end_positions.get(section["node_id"])

            if position < start_position:
                continue

            if end_position is not None and position >= end_position:
                continue

            candidates.append(section)

        if not candidates:
            unclassified.append(record)
            continue

        owner = max(
            candidates,
            key=lambda item: (
                item["level"],
                item["start_page"],
                item["start_block"],
            ),
        )

        assigned[owner["node_id"]].append(record)

    return assigned, unclassified


def _populate_section_content(
    sections: list[dict],
    assigned: dict,
) -> None:
    """Populate section text and page children."""
    for section in sections:
        section_records = assigned.get(section["node_id"], [])

        section_records.sort(key=_position_key)

        section["content"] = "\n".join(
            record["text"]
            for record in section_records
        ).strip()

        by_page = {}

        for record in section_records:
            by_page.setdefault(
                record["page_number"], []
            ).append(record)

        section["children"].extend(
            _make_page_node(
                page_number=page_number,
                blocks=page_records,
                document_id=section["node_id"],
            )
            for page_number, page_records in sorted(by_page.items())
        )


def _make_unclassified_node(
    document_id: str,
    records: list[dict],
) -> dict | None:
    """
    Preserve text that cannot safely be assigned to a section.

    This includes front matter, abstracts, metadata, and content
    preceding the first recognized heading.
    """
    if not records:
        return None

    records = sorted(records, key=_position_key)

    by_page = {}

    for record in records:
        by_page.setdefault(
            record["page_number"], []
        ).append(record)

    children = [
        _make_page_node(
            page_number=page_number,
            blocks=page_records,
            document_id=f"{document_id}_unclassified",
        )
        for page_number, page_records in sorted(by_page.items())
    ]

    first_page = min(
        record["page_number"]
        for record in records
    )

    first_page_records = [
        record
        for record in records
        if record["page_number"] == first_page
    ]

    return {
        "node_id": f"{document_id}_unclassified",
        "node_type": "section",
        "title": "Unclassified content",
        "level": 1,
        "start_page": first_page,
        "start_block": min(
            record["block_index"]
            for record in first_page_records
        ),
        "end_page": max(
            record["page_number"]
            for record in records
        ),
        "summary": "",
        "content": "\n".join(
            record["text"] for record in records
        ).strip(),
        "children": children,
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
    Build a document tree with nested sections and page nodes.

    Every extracted text block is assigned either to a detected
    section or to an Unclassified content node.
    """
    page_numbers = sorted({
        page["page_number"]
        for page in pages
    })

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

    records = sorted(records, key=_position_key)
    outline = _deduplicate_outline(outline)

    if not outline:
        outline = [{
            "title": "Document content",
            "level": 1,
            "start_page": page_numbers[0],
            "start_block": 0,
        }]

    sections, roots = _attach_sections(
        outline,
        document_id,
    )

    end_positions = _calculate_section_end_positions(
        sections,
        page_numbers[-1],
    )

    assigned, unclassified = _assign_records_to_sections(
        records,
        sections,
        end_positions,
    )

    _populate_section_content(sections, assigned)

    unclassified_node = _make_unclassified_node(
        document_id,
        unclassified,
    )

    if unclassified_node:
        roots.insert(0, unclassified_node)

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
    Build an adaptive hierarchical index for an extracted PDF.

    Processing flow:
        1. Prepare text and layout metadata.
        2. Split outline inference into smaller Groq requests.
        3. Validate headings against actual document blocks.
        4. Use local heading detection if AI inference fails
           or returns no usable headings.
        5. Build nested section and page nodes.
        6. Preserve unmatched content separately.

    The public function signature remains unchanged.
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

    if not outline:
        outline = _fallback_outline(records)

    return _build_hierarchy(
        document_id=document_id,
        filename=filename,
        document_title=document_title,
        pages=prepared_pages,
        records=records,
        outline=outline,
    )
