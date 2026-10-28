
import re

from database import (
    page_index_trees_collection,
    pages_collection,
)
from services.llm_service import understand_query

def _normalize(text: str) -> str:
    """Normalize text for keyword matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokenize(text: str) -> set[str]:
    """Convert text into searchable words."""
    return set(_normalize(text).split())


def _score_text(text: str, query: dict) -> int:
    """Score text using query topic, keywords, and information needed."""
    normalized_text = _normalize(text)
    text_tokens = _tokenize(normalized_text)

    query_tokens = _tokenize(
        f"{query.get('topic', '')} "
        f"{query.get('information_needed', '')}"
    )

    score = len(text_tokens & query_tokens) * 2

    for keyword in query.get("keywords", []):
        normalized_keyword = _normalize(keyword)

        if not normalized_keyword:
            continue

        if normalized_keyword in normalized_text:
            score += 5

        keyword_tokens = _tokenize(normalized_keyword)
        score += len(keyword_tokens & text_tokens)

    return score


def _score_section(section: dict, query: dict) -> int:
    """Score a section using its title and summary."""
    title = section.get("title", "")
    summary = section.get("summary", "")

    return (
        _score_text(title, query) * 2
        + _score_text(summary, query)
    )


def retrieve_relevant_pages(
    question: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> dict:
    """
    Rank sections and individual pages, then return only the most
    relevant pages with their section metadata.

    top_k is the maximum number of pages returned, not sections.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_analysis = understand_query(question)

    tree_filter = {"document_id": document_id} if document_id else {}
    tree_cursor = page_index_trees_collection.find(tree_filter)

    ranked_sections = []

    for tree in tree_cursor:
        for section in tree.get("children", []):
            if section.get("node_type") != "section":
                continue

            start_page = section.get("start_page")
            end_page = section.get("end_page")

            if start_page is None or end_page is None:
                continue

            section_score = _score_section(section, query_analysis)

            ranked_sections.append({
                "document_id": tree["document_id"],
                "filename": tree.get("filename", ""),
                "section_id": section.get("node_id"),
                "section_title": section.get("title", ""),
                "start_page": start_page,
                "end_page": end_page,
                "section_summary": section.get("summary", ""),
                "section_score": section_score,
            })

    # Skip sections with no lexical match.
    ranked_sections = [
        section for section in ranked_sections
        if section["section_score"] > 0
    ]

    ranked_sections.sort(
        key=lambda section: section["section_score"],
        reverse=True,
    )

    ranked_pages = []

    for section in ranked_sections:
        page_filter = {
            "document_id": section["document_id"],
            "page_number": {
                "$gte": section["start_page"],
                "$lte": section["end_page"],
            },
        }

        page_cursor = pages_collection.find(
            page_filter,
            {
                "_id": 0,
                "document_id": 1,
                "filename": 1,
                "page_number": 1,
                "text": 1,
            },
        )

        for page in page_cursor:
            page_text = (page.get("text") or "").strip()

            if not page_text:
                continue

            page_score = _score_text(page_text, query_analysis)

            combined_score = page_score + section["section_score"]

            if combined_score <= 0:
                continue

            ranked_pages.append({
                **section,
                "page_number": page["page_number"],
                "text": page_text,
                "relevance_score": combined_score,
            })

    unique_pages = {}

    for page in ranked_pages:
        key = (
            str(page["document_id"]),
            page["page_number"],
        )

        if (
            key not in unique_pages
            or page["relevance_score"]
            > unique_pages[key]["relevance_score"]
        ):
            unique_pages[key] = page

    ranked_pages = sorted(
        unique_pages.values(),
        key=lambda page: page["relevance_score"],
        reverse=True,
    )

    selected_pages = ranked_pages[:top_k]

    grouped_results = []
    groups = {}

    for page in selected_pages:
        key = (
            str(page["document_id"]),
            page["section_id"],
        )

        if key not in groups:
            group = {
                "document_id": page["document_id"],
                "filename": page["filename"],
                "section_id": page["section_id"],
                "section_title": page["section_title"],
                "start_page": page["start_page"],
                "end_page": page["end_page"],
                "section_summary": page["section_summary"],
                "relevance_score": page["relevance_score"],
                "pages": [],
            }

            groups[key] = group
            grouped_results.append(group)

        groups[key]["pages"].append({
            "document_id": page["document_id"],
            "filename": page["filename"],
            "page_number": page["page_number"],
            "text": page["text"],
            "relevance_score": page["relevance_score"],
        })

    return {
        "question": question,
        "query_analysis": query_analysis,
        "sections_found": len(grouped_results),
        "pages_found": len(selected_pages),
        "results": grouped_results,
    }
