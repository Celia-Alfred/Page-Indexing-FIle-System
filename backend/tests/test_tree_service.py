
import pytest

import services.tree_service as tree_service
from services.tree_service import build_page_index_tree


@pytest.fixture
def sample_pages():
    """Sample pages containing text and heading-format metadata."""
    return [
        {
            "page_number": 1,
            "blocks": [
                {
                    "text": "Introduction",
                    "lines": [
                        {
                            "text": "Introduction",
                            "spans": [
                                {
                                    "text": "Introduction",
                                    "font_size": 18,
                                    "font": "Arial-Bold",
                                    "flags": 16,
                                }
                            ],
                        }
                    ],
                },
                {"text": "This document introduces artificial intelligence."},
                {"text": "1.1 Background"},
                {"text": "AI systems learn patterns from data."},
            ],
        },
        {
            "page_number": 2,
            "blocks": [
                {"text": "Methods"},
                {"text": "The method evaluates several model configurations."},
            ],
        },
    ]


def fake_outline_response(document_title, headings):
    """Create a mock Gemini response containing JSON."""
    import json
    from unittest.mock import Mock

    response = Mock()
    response.text = json.dumps(
        {
            "document_title": document_title,
            "headings": headings,
        }
    )
    return response


def test_build_tree_has_document_metadata(sample_pages, monkeypatch):
    """The tree root should contain the document metadata."""
    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        lambda *args, **kwargs: fake_outline_response(
            "AI Research",
            [
                {
                    "title": "Introduction",
                    "level": 1,
                    "start_page": 1,
                    "start_block": 0,
                },
                {
                    "title": "Methods",
                    "level": 1,
                    "start_page": 2,
                    "start_block": 0,
                },
            ],
        ),
    )

    tree = build_page_index_tree(
        document_id="test-doc-001",
        filename="research.pdf",
        pages=sample_pages,
    )

    assert tree["node_id"] == "test-doc-001"
    assert tree["node_type"] == "document"
    assert tree["filename"] == "research.pdf"
    assert tree["title"] == "AI Research"
    assert tree["start_page"] == 1
    assert tree["end_page"] == 2
    assert isinstance(tree["children"], list)


def test_tree_contains_nested_sections_and_page_nodes(
    sample_pages, monkeypatch
):
    """Sections should contain page nodes, with nested levels preserved."""
    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        lambda *args, **kwargs: fake_outline_response(
            "AI Research",
            [
                {
                    "title": "Introduction",
                    "level": 1,
                    "start_page": 1,
                    "start_block": 0,
                },
                {
                    "title": "1.1 Background",
                    "level": 2,
                    "start_page": 1,
                    "start_block": 2,
                },
                {
                    "title": "Methods",
                    "level": 1,
                    "start_page": 2,
                    "start_block": 0,
                },
            ],
        ),
    )

    tree = build_page_index_tree(
        document_id="test-doc-002",
        filename="research.pdf",
        pages=sample_pages,
    )

    introduction = next(
        node
        for node in tree["children"]
        if node["title"] == "Introduction"
    )

    background = next(
        node
        for node in introduction["children"]
        if node["node_type"] == "section"
    )

    assert background["title"] == "1.1 Background"
    assert background["level"] == 2

    page_nodes = [
        node
        for node in background["children"]
        if node["node_type"] == "page"
    ]

    assert page_nodes
    assert all("page_number" in node for node in page_nodes)


def test_page_content_is_assigned_to_sections(sample_pages, monkeypatch):
    """Each section should contain the text assigned to its heading range."""
    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        lambda *args, **kwargs: fake_outline_response(
            "AI Research",
            [
                {
                    "title": "Introduction",
                    "level": 1,
                    "start_page": 1,
                    "start_block": 0,
                },
                {
                    "title": "Methods",
                    "level": 1,
                    "start_page": 2,
                    "start_block": 0,
                },
            ],
        ),
    )

    tree = build_page_index_tree(
        document_id="test-doc-003",
        filename="research.pdf",
        pages=sample_pages,
    )

    introduction = next(
        node
        for node in tree["children"]
        if node["title"] == "Introduction"
    )
    methods = next(
        node
        for node in tree["children"]
        if node["title"] == "Methods"
    )

    assert "Introduction" in introduction["content"]
    assert "artificial intelligence" in introduction["content"]
    assert "model configurations" in methods["content"]

    # The method section should not contain introduction-page content.
    assert "artificial intelligence" not in methods["content"]


def test_gemini_failure_uses_fallback_outline(
    sample_pages, monkeypatch
):
    """A failed outline request should trigger the local fallback."""
    def raise_error(*args, **kwargs):
        raise RuntimeError("Simulated Gemini failure")

    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        raise_error,
    )

    tree = build_page_index_tree(
        document_id="test-doc-004",
        filename="research.pdf",
        pages=sample_pages,
    )

    assert tree["node_type"] == "document"
    assert tree["children"]
    assert any(
        node["node_type"] == "section"
        for node in tree["children"]
    )


def test_empty_pages_return_warning():
    """A document with no extractable text should return a warning."""
    tree = build_page_index_tree(
        document_id="test-doc-005",
        filename="empty.pdf",
        pages=[
            {"page_number": 1, "blocks": []},
            {"page_number": 2, "text": "   "},
        ],
    )

    assert tree["node_type"] == "document"
    assert tree["children"] == []
    assert "warning" in tree


def test_page_text_fallback_without_blocks(monkeypatch):
    """Page-level text should be retained when block metadata is absent."""
    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        lambda *args, **kwargs: fake_outline_response(
            "Plain Text Document",
            [],
        ),
    )

    tree = build_page_index_tree(
        document_id="test-doc-006",
        filename="plain.pdf",
        pages=[
            {
                "page_number": 1,
                "text": "This page contains readable text.",
            }
        ],
    )

    section = tree["children"][0]

    assert section["title"] == "Document content"
    assert "readable text" in section["content"]
    assert section["children"][0]["node_type"] == "page"


def test_invalid_gemini_heading_positions_are_ignored(
    sample_pages, monkeypatch
):
    """Gemini headings must refer to real page/block positions."""
    monkeypatch.setattr(
        tree_service,
        "generate_content_with_retry",
        lambda *args, **kwargs: fake_outline_response(
            "AI Research",
            [
                {
                    "title": "Valid Heading",
                    "level": 1,
                    "start_page": 1,
                    "start_block": 0,
                },
                {
                    "title": "Invalid Heading",
                    "level": 1,
                    "start_page": 99,
                    "start_block": 0,
                },
            ],
        ),
    )

    tree = build_page_index_tree(
        document_id="test-doc-007",
        filename="research.pdf",
        pages=sample_pages,
    )

    titles = [node["title"] for node in tree["children"]]

    assert "Valid Heading" in titles
    assert "Invalid Heading" not in titles
