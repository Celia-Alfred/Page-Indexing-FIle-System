
import pymupdf


def reconstruct_line_text(spans: list[dict]) -> str:
    """Reconstruct line text while preserving likely word spaces."""

    if not spans:
        return ""

    ordered_spans = sorted(
        spans,
        key=lambda span: (
            span["bbox"][0]
            if span.get("bbox")
            else 0
        ),
    )

    parts = []
    previous_span = None

    for span in ordered_spans:
        text = span.get("text", "")

        if not text:
            continue

        if previous_span is not None:
            previous_bbox = previous_span.get("bbox")
            current_bbox = span.get("bbox")

            if previous_bbox and current_bbox:
                gap = current_bbox[0] - previous_bbox[2]

                font_size = max(
                    previous_span.get("font_size", 0),
                    span.get("font_size", 0),
                )

                # Add a space when the horizontal gap suggests
                # that the PDF contains separate words.
                if gap > font_size * 0.15:
                    parts.append(" ")

        parts.append(text)
        previous_span = span

    return "".join(parts)


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract page text and layout metadata from a PDF."""

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:
        for page in pdf:
            page_data = page.get_text("dict")
            blocks = []

            for block in page_data.get("blocks", []):
                if block.get("type") != 0:
                    continue

                lines = []

                for line in block.get("lines", []):
                    spans = [
                        {
                            "text": span.get("text", ""),
                            "font": span.get("font", ""),
                            "font_size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "bbox": span.get("bbox"),
                        }
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ]

                    if spans:
                        lines.append({
                            "text": reconstruct_line_text(spans),
                            "bbox": line.get("bbox"),
                            "spans": spans,
                        })

                if lines:
                    blocks.append({
                        "bbox": block.get("bbox"),
                        "lines": lines,
                        "text": "\n".join(
                            line["text"] for line in lines
                        ),
                    })

            pages.append({
                "page_number": page.number + 1,
                "text": page.get_text().strip(),
                "blocks": blocks,
            })

    return pages


def extract_pdf_toc(pdf_bytes: bytes) -> list[dict]:
    """Extract the PDF's embedded table of contents, if available."""

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:
        return [
            {
                "level": level,
                "title": title.strip(),
                "page_number": page_number,
            }
            for level, title, page_number, *_ in pdf.get_toc()
            if title.strip() and page_number >= 1
        ]
