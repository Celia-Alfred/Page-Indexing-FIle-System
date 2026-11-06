
from unittest.mock import patch

from services.tree_service import build_page_index_tree


@patch("services.tree_service._infer_outline")
def test_build_page_index_tree(mock_outline):
    mock_outline.return_value = (
        "AWS Concepts",
        [
            {
                "title": "Introduction",
                "level": 1,
                "start_page": 1,
                "start_block": 0,
            }
        ],
    )

    document_id = "test-doc-123"
    filename = "sample.pdf"

    pages = [
        {"page_number": 1, "text": "Introduction to AWS."},
        {"page_number": 2, "text": "AWS architecture concepts."},
    ]

    tree = build_page_index_tree(document_id, filename, pages)

    assert tree["node_id"] == document_id
    assert tree["filename"] == filename
    assert tree["node_type"] == "document"
    assert tree["title"] == "AWS Concepts"
    assert len(tree["children"]) == 1

    section = tree["children"][0]

    assert section["node_type"] == "section"
    assert section["title"] == "Introduction"
    assert section["start_page"] == 1
    assert section["end_page"] == 2
    assert len(section["children"]) == 2

    assert section["children"][0]["page_number"] == 1
    assert section["children"][0]["node_type"] == "page"

    mock_outline.assert_called_once()


@patch("services.tree_service._infer_outline")
def test_build_page_index_tree_creates_multiple_sections(mock_outline):
    mock_outline.return_value = (
        "Sample Document",
        [
            {
                "title": "Section One",
                "level": 1,
                "start_page": 1,
                "start_block": 0,
            },
            {
                "title": "Section Two",
                "level": 1,
                "start_page": 4,
                "start_block": 0,
            },
            {
                "title": "Section Three",
                "level": 1,
                "start_page": 7,
                "start_block": 0,
            },
        ],
    )

    pages = [
        {"page_number": number, "text": f"Content of page {number}"}
        for number in range(1, 8)
    ]

    tree = build_page_index_tree(
        "test-doc-456",
        "sample.pdf",
        pages,
    )

    sections = tree["children"]

    assert len(sections) == 3

    assert len(sections[0]["children"]) == 3
    assert len(sections[1]["children"]) == 3
    assert len(sections[2]["children"]) == 1

    assert sections[0]["title"] == "Section One"
    assert sections[1]["title"] == "Section Two"
    assert sections[2]["title"] == "Section Three"

    mock_outline.assert_called_once()


def test_build_page_index_tree_with_no_pages():
    tree = build_page_index_tree(
        "test-doc-789",
        "empty.pdf",
        [],
    )

    assert tree["node_id"] == "test-doc-789"
    assert tree["children"] == []
    assert tree["warning"] == "No extractable text was found in this PDF."
