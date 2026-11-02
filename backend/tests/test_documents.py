
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.answer_service as answer_service


def make_words(text):
    """Create predictable word-level PDF coordinates for tests."""
    words = []

    for index, token in enumerate(text.split()):
        words.append({
            "text": token,
            "bbox": [
                50 + index * 45,
                100,
                90 + index * 45,
                115,
            ],
            "block_no": 0,
            "line_no": 0,
            "word_no": index,
        })

    return words


def make_gemini_response(
    answer="Machine learning uses data to train models [S1].",
    evidence=None,
):
    """Create a Gemini-style JSON response."""
    if evidence is None:
        evidence = [
            {
                "source_id": "S1",
                "quote": "Machine learning uses data to train models.",
            }
        ]

    return SimpleNamespace(
        text=json.dumps({
            "answer": answer,
            "evidence": evidence,
        })
    )


@pytest.fixture
def mock_gemini(monkeypatch):
    """Mock Gemini without making real API calls."""
    generate_mock = MagicMock(
        return_value=make_gemini_response()
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        generate_mock,
    )

    return generate_mock


@pytest.fixture
def retrieved_sections():
    first_text = (
        "Machine learning uses data "
        "to train models."
    )

    second_text = "Training requires suitable data."

    return [
        {
            "document_id": "doc-123",
            "filename": "research.pdf",
            "section_title": "Machine Learning",
            "relevance_score": 1,
            "pages": [
                {
                    "document_id": "doc-123",
                    "filename": "research.pdf",
                    "page_number": 3,
                    "text": first_text,
                    "words": make_words(first_text),
                    "page_width": 595,
                    "page_height": 842,
                    "page_rotation": 0,
                    "coordinate_system": "pymupdf_unrotated",
                    "relevance_score": 1,
                },
                {
                    "document_id": "doc-123",
                    "filename": "research.pdf",
                    "page_number": 4,
                    "text": second_text,
                    "words": make_words(second_text),
                    "page_width": 595,
                    "page_height": 842,
                    "page_rotation": 0,
                    "coordinate_system": "pymupdf_unrotated",
                    "relevance_score": 1,
                },
            ],
        }
    ]


def test_generate_answer_returns_grounded_answer(
    mock_gemini,
    retrieved_sections,
):
    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert result["answer"] == (
        "Machine learning uses data to train models [S1]."
    )

    assert len(result["citations"]) == 1

    citation = result["citations"][0]

    assert citation["source_id"] == "S1"
    assert citation["filename"] == "research.pdf"
    assert citation["page_number"] == 3

    assert citation["excerpt"] == (
        "Machine learning uses data to train models."
    )

    assert len(citation["evidence"]) == 1

    assert citation["evidence"][0][
        "highlight_rectangles"
    ]

    mock_gemini.assert_called_once()


def test_prompt_contains_question_and_page_context(
    mock_gemini,
    retrieved_sections,
):
    answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    prompt = mock_gemini.call_args.args[0]

    assert "Explain machine learning" in prompt
    assert "research.pdf" in prompt
    assert "Page: 3" in prompt
    assert "Machine learning uses data" in prompt
    assert "[S1]" in prompt
    assert "exact supporting quotation" in prompt.lower()


def test_only_cited_sources_are_returned(
    monkeypatch,
    retrieved_sections,
):
    response = make_gemini_response(
        answer="The answer uses the second page [S2].",
        evidence=[
            {
                "source_id": "S2",
                "quote": "Training requires suitable data.",
            }
        ],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "What is required for training?",
        retrieved_sections,
    )

    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_id"] == "S2"
    assert result["citations"][0]["page_number"] == 4


def test_invalid_citation_ids_are_removed(
    monkeypatch,
    retrieved_sections,
):
    response = make_gemini_response(
        answer="The answer is supported [S1] and [S99].",
        evidence=[
            {
                "source_id": "S1",
                "quote": "Machine learning uses data to train models.",
            }
        ],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert "[S1]" in result["answer"]
    assert "[S99]" not in result["answer"]

    assert [
        citation["source_id"]
        for citation in result["citations"]
    ] == ["S1"]


def test_duplicate_pages_are_cited_only_once(
    monkeypatch,
):
    text = "Same page content."

    response = make_gemini_response(
        answer="See the source [S1].",
        evidence=[
            {
                "source_id": "S1",
                "quote": "Same page content.",
            }
        ],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    sections = [
        {
            "document_id": "doc-123",
            "filename": "research.pdf",
            "section_title": "Section A",
            "pages": [
                {
                    "page_number": 1,
                    "text": text,
                    "words": make_words(text),
                }
            ],
        },
        {
            "document_id": "doc-123",
            "filename": "research.pdf",
            "section_title": "Section B",
            "pages": [
                {
                    "page_number": 1,
                    "text": text,
                    "words": make_words(text),
                }
            ],
        },
    ]

    result = answer_service.generate_answer(
        "What does the page say?",
        sections,
    )

    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 1


def test_empty_question_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        answer_service.generate_answer("   ", [])


def test_no_retrieved_sections_returns_no_results_message(
    mock_gemini,
):
    result = answer_service.generate_answer(
        "Explain machine learning",
        [],
    )

    assert "couldn't find relevant information" in result["answer"]
    assert result["citations"] == []

    mock_gemini.assert_not_called()


def test_sections_without_readable_text_return_no_citations(
    mock_gemini,
):
    sections = [
        {
            "document_id": "doc-123",
            "filename": "empty.pdf",
            "section_title": "Empty section",
            "pages": [
                {
                    "page_number": 1,
                    "text": "   ",
                }
            ],
        }
    ]

    result = answer_service.generate_answer(
        "What is in this document?",
        sections,
    )

    assert "no readable page text" in result["answer"]
    assert result["citations"] == []

    mock_gemini.assert_not_called()


def test_empty_gemini_response_raises_runtime_error(
    monkeypatch,
    retrieved_sections,
):
    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: SimpleNamespace(text="  "),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini returned an empty answer",
    ):
        answer_service.generate_answer(
            "Explain machine learning",
            retrieved_sections,
        )


def test_multiple_citations_are_returned_in_source_order(
    monkeypatch,
    retrieved_sections,
):
    response = make_gemini_response(
        answer="First claim [S1]. Second claim [S2].",
        evidence=[
            {
                "source_id": "S2",
                "quote": "Training requires suitable data.",
            },
            {
                "source_id": "S1",
                "quote": "Machine learning uses data to train models.",
            },
        ],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert [
        citation["source_id"]
        for citation in result["citations"]
    ] == ["S1", "S2"]


def test_unverified_quote_does_not_produce_citation(
    monkeypatch,
    retrieved_sections,
):
    response = make_gemini_response(
        answer="Machine learning uses data [S1].",
        evidence=[
            {
                "source_id": "S1",
                "quote": "Machine learning is powered by magic.",
            }
        ],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert "[S1]" not in result["answer"]
    assert result["citations"] == []


def test_missing_word_coordinates_do_not_produce_highlight(
    monkeypatch,
    retrieved_sections,
):
    retrieved_sections[0]["pages"][0]["words"] = []

    response = make_gemini_response()

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert "[S1]" not in result["answer"]
    assert result["citations"] == []


def test_highlight_rectangles_are_generated_per_line(
    monkeypatch,
    retrieved_sections,
):
    text = "Machine learning uses data to train models."

    words = make_words(text)

    # Simulate the final two words appearing on another line.
    words[-2]["line_no"] = 1
    words[-1]["line_no"] = 1
    words[-2]["bbox"] = [50, 120, 90, 135]
    words[-1]["bbox"] = [95, 120, 135, 135]

    retrieved_sections[0]["pages"][0]["words"] = words

    response = make_gemini_response()

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    rectangles = result["citations"][0]["evidence"][0][
        "highlight_rectangles"
    ]

    assert len(rectangles) == 2
    assert rectangles[0][1] == 100
    assert rectangles[1][1] == 120


def test_invalid_word_bbox_does_not_produce_highlight(
    monkeypatch,
    retrieved_sections,
):
    words = make_words(
        "Machine learning uses data to train models."
    )

    words[0]["bbox"] = [100, 100, 50, 115]

    retrieved_sections[0]["pages"][0]["words"] = words

    response = make_gemini_response()

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert "[S1]" not in result["answer"]
    assert result["citations"] == []


def test_duplicate_evidence_quotes_are_deduplicated(
    monkeypatch,
    retrieved_sections,
):
    quote = "Machine learning uses data to train models."

    response = make_gemini_response(
        evidence=[
            {
                "source_id": "S1",
                "quote": quote,
            },
            {
                "source_id": "S1",
                "quote": quote,
            },
        ]
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert len(result["citations"]) == 1
    assert len(result["citations"][0]["evidence"]) == 1


def test_generate_content_is_called_with_prompt_only(
    mock_gemini,
    retrieved_sections,
):
    answer_service.generate_answer(
        "Explain machine learning",
        retrieved_sections,
    )

    assert mock_gemini.call_count == 1
    assert len(mock_gemini.call_args.args) == 1
    assert mock_gemini.call_args.kwargs == {}


def test_invalid_json_response_raises_runtime_error(
    monkeypatch,
    retrieved_sections,
):
    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: SimpleNamespace(
            text="This is not JSON"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini returned invalid JSON",
    ):
        answer_service.generate_answer(
            "Explain machine learning",
            retrieved_sections,
        )


def test_answer_without_evidence_has_no_citations(
    monkeypatch,
    retrieved_sections,
):
    response = make_gemini_response(
        answer="The documents do not contain this information.",
        evidence=[],
    )

    monkeypatch.setattr(
        answer_service,
        "generate_content_with_retry",
        lambda prompt: response,
    )

    result = answer_service.generate_answer(
        "What is the weather?",
        retrieved_sections,
    )

    assert result["citations"] == []
