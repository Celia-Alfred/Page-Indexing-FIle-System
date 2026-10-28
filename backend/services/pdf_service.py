
import pymupdf


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract text and page numbers from a PDF."""
    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in pdf:
            pages.append({
                "page_number": page.number + 1,
                "text": page.get_text().strip(),
            })

    return pages
