
from unittest.mock import patch

from services.tree_service import build_page_index_tree


@patch("services.tree_service.generate_summary")
def test_build_page_index_tree(mock_summary):
    mock_summary.return_value = "Summary of AWS concepts."

    document_id = "test-doc-123"
    filename = "sample.pdf"

    pages = [
        {"page_number": 1, "text": "Introduction to AWS."},
        {"page_number": 2, "text": "AWS architecture concepts."},
    ]

    tree = build_page_index_tree(document_id, filename, pages)

    assert tree["document_id"] == document_id
    assert tree["filename"] == filename
    assert tree["node_type"] == "document"
    assert len(tree["children"]) == 1

    section = tree["children"][0]

    assert section["node_type"] == "section"
    assert section["start_page"] == 1
    assert section["end_page"] == 2
    assert section["summary"] == "Summary of AWS concepts."
    assert len(section["children"]) == 2

    assert section["children"][0]["page_number"] == 1
    assert section["children"][0]["node_type"] == "page"

    mock_summary.assert_called_once()


@patch("services.tree_service.generate_summary")
def test_build_page_index_tree_creates_multiple_sections(mock_summary):
    mock_summary.return_value = "Section summary."

    pages = [
        {"page_number": number, "text": f"Content of page {number}"}
        for number in range(1, 8)
    ]

    tree = build_page_index_tree(
        "test-doc-456",
        "sample.pdf",
        pages,
        pages_per_section=3,
    )

    sections = tree["children"]

    assert len(sections) == 3
    assert len(sections[0]["children"]) == 3
    assert len(sections[1]["children"]) == 3
    assert len(sections[2]["children"]) == 1

    assert mock_summary.call_count == 3


def test_build_page_index_tree_with_no_pages():
    tree = build_page_index_tree(
        "test-doc-789",
        "empty.pdf",
        [],
    )

    assert tree["children"] == []
