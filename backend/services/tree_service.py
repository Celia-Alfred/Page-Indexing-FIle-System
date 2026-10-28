
from services.llm_service import generate_summary

def build_page_index_tree(
    document_id: str,
    filename: str,
    pages: list[dict],
    pages_per_section: int = 5,
) -> dict:
    """Build a hierarchical document tree with Gemini-generated summaries."""

    if pages_per_section < 1:
        raise ValueError("pages_per_section must be at least 1")

    section_nodes = []

    for start in range(0, len(pages), pages_per_section):
        section_pages = pages[start:start + pages_per_section]

        first_page = section_pages[0]["page_number"]
        last_page = section_pages[-1]["page_number"]

        section_text = "\n\n".join(
            page["text"].strip()
            for page in section_pages
            if page["text"].strip()
        )

        if section_text:
            section_summary = generate_summary(section_text)
        else:
            section_summary = "No readable text found in this section."

        page_nodes = [
            {
                "node_id": (
                    f"{document_id}_page_{page['page_number']}"
                ),
                "node_type": "page",
                "page_number": page["page_number"],
                "summary": page["text"][:300],
                "children": [],
            }
            for page in section_pages
        ]

        section_nodes.append({
            "node_id": (
                f"{document_id}_section_{len(section_nodes) + 1}"
            ),
            "node_type": "section",
            "title": f"Pages {first_page}-{last_page}",
            "start_page": first_page,
            "end_page": last_page,
            "summary": section_summary,
            "children": page_nodes,
        })

    return {
        "document_id": document_id,
        "filename": filename,
        "node_type": "document",
        "title": filename,
        "children": section_nodes,
    }
