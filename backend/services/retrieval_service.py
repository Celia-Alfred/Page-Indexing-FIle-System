
import json
import logging
from typing import Any

from database import (
    page_index_trees_collection,
    pages_collection,
)
from services.llm_service import (
    understand_query,
    generate_content_with_retry,
)

logger = logging.getLogger(__name__)

MAX_SEARCH_ROUNDS = 5
MAX_SECTIONS_PER_ROUND = 3
MAX_TREE_NODES_IN_PROMPT = 160
MAX_PAGES_IN_CONTEXT = 12
MAX_PAGE_CHARS = 9000


def _parse_json_response(response: Any) -> dict:
    """Parse and validate a Gemini JSON response."""
    response_text = (
        getattr(response, "text", None) or ""
    ).strip()

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty retrieval response"
        )

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON during tree retrieval"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Gemini returned an unexpected JSON structure"
        )

    return result


def _valid_page_range(node: dict) -> tuple[int, int] | None:
    """Return a valid inclusive page range."""
    start = node.get("start_page")
    end = node.get("end_page")

    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 1
        or end < start
    ):
        return None

    return start, end


def _collect_sections(tree: dict) -> list[dict]:
    """Flatten the tree while preserving its hierarchy."""
    sections = []

    def visit(nodes, ancestors=None):
        if ancestors is None:
            ancestors = []

        if not isinstance(nodes, list):
            return

        for node in nodes:
            if not isinstance(node, dict):
                continue

            next_ancestors = ancestors

            if node.get("node_type") == "section":
                title = str(node.get("title") or "").strip()
                page_range = _valid_page_range(node)
                node_id = node.get("node_id")

                if node_id and page_range:
                    sections.append({
                        "node_id": str(node_id),
                        "title": title,
                        "ancestor_titles": list(ancestors),
                        "depth": len(ancestors),
                        "start_page": page_range[0],
                        "end_page": page_range[1],
                    })

                next_ancestors = (
                    ancestors + [title]
                    if title
                    else ancestors
                )

            visit(
                node.get("children", []),
                next_ancestors,
            )

    visit(tree.get("children", []))

    unique_sections = {}

    for section in sections:
        unique_sections.setdefault(
            section["node_id"],
            section,
        )

    return list(unique_sections.values())


def _make_tree_outline(sections: list[dict]) -> list[dict]:
    """Create compact metadata for reasoning over the document tree."""
    return [
        {
            "node_id": section["node_id"],
            "title": section["title"],
            "parent_path": " > ".join(
                section["ancestor_titles"]
            ),
            "depth": section["depth"],
            "start_page": section["start_page"],
            "end_page": section["end_page"],
        }
        for section in sections[:MAX_TREE_NODES_IN_PROMPT]
    ]


def _select_sections(
    question: str,
    query_analysis: dict,
    outline: list[dict],
    visited_ids: set[str],
    previous_evidence: list[dict],
) -> list[str]:
    """Ask Gemini to select useful sections from the document tree."""
    remaining = [
        node
        for node in outline
        if node["node_id"] not in visited_ids
    ]

    if not remaining:
        return []

    evidence_summary = [
        {
            "filename": item["filename"],
            "page_number": item["page_number"],
            "section_title": item["section_title"],
            "text": item["text"][:1200],
        }
        for item in previous_evidence[-6:]
    ]

    prompt = f"""
You are the tree-navigation agent of a vectorless document QA system.

Navigate the document's hierarchical section tree to find evidence
that answers the user's question.

Reason about section meaning, parent hierarchy, page ranges,
and evidence already inspected.

Do not use embeddings, vector search, or keyword-overlap scores.

QUESTION:
{question}

QUESTION ANALYSIS:
{json.dumps(query_analysis, ensure_ascii=False)}

AVAILABLE TREE SECTIONS:
{json.dumps(remaining, ensure_ascii=False)}

EVIDENCE ALREADY INSPECTED:
{json.dumps(evidence_summary, ensure_ascii=False)}

Return valid JSON with exactly these fields:
{{
  "selected_node_ids": ["existing node_id"],
  "reasoning": "Brief explanation of the selection"
}}

Rules:
- Select at most {MAX_SECTIONS_PER_ROUND} sections.
- Use only node_ids from AVAILABLE TREE SECTIONS.
- Prefer specific subsections when appropriate.
- Follow relevant references, definitions, related topics,
  and dependencies when they may contain the answer.
- Do not select previously inspected nodes.
- If no remaining section is useful, return an empty list.
- Return JSON only, without Markdown fences.
"""

    response = generate_content_with_retry(
        prompt,
        config={"response_mime_type": "application/json"},
    )

    result = _parse_json_response(response)
    selected = result.get("selected_node_ids", [])

    if not isinstance(selected, list):
        return []

    valid_ids = {
        node["node_id"]
        for node in remaining
    }

    return list(dict.fromkeys(
        node_id
        for node_id in selected
        if isinstance(node_id, str)
        and node_id in valid_ids
    ))[:MAX_SECTIONS_PER_ROUND]


def _read_section_pages(
    document_id: Any,
    filename: str,
    selected_sections: list[dict],
) -> list[dict]:
    """
    Read original page text, extracted blocks, and coordinate metadata
    covered by the selected sections.
    """
    if not selected_sections:
        return []

    pages_cursor = pages_collection.find(
        {"document_id": document_id},
        {
            "_id": 0,
            "document_id": 1,
            "filename": 1,
            "page_number": 1,
            "text": 1,

            "blocks": 1,

            "words": 1,

            "page_width": 1,
            "page_height": 1,
            "page_rotation": 1,
            "coordinate_system": 1,
        },
    )

    section_by_page = {}

    for page in pages_cursor:
        page_number = page.get("page_number")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
        ):
            continue

        matching_sections = [
            section
            for section in selected_sections
            if (
                section["start_page"]
                <= page_number
                <= section["end_page"]
            )
        ]

        if not matching_sections:
            continue

        matching_sections.sort(
            key=lambda section: (
                section["depth"],
                -(
                    section["end_page"]
                    - section["start_page"]
                ),
            ),
            reverse=True,
        )

        section_by_page[page_number] = (
            matching_sections[0],
            page,
        )

    results = []

    for page_number in sorted(section_by_page):
        section, page = section_by_page[page_number]
        text = (page.get("text") or "").strip()

        if not text:
            continue

        results.append({
            "document_id": page.get(
                "document_id",
                document_id,
            ),
            "filename": page.get("filename") or filename,
            "page_number": page_number,
            "text": text[:MAX_PAGE_CHARS],

            "blocks": page.get("blocks", []),

            "words": page.get("words", []),

            "page_width": page.get("page_width"),
            "page_height": page.get("page_height"),
            "page_rotation": page.get("page_rotation", 0),
            "coordinate_system": page.get(
                "coordinate_system",
                "pymupdf_unrotated",
            ),

            "section_id": section["node_id"],
            "section_title": section["title"],
            "ancestor_titles": section["ancestor_titles"],
            "start_page": section["start_page"],
            "end_page": section["end_page"],
        })

    return results


def _assess_evidence(
    question: str,
    evidence: list[dict],
    visited_ids: set[str],
    outline: list[dict],
) -> dict:
    """Ask Gemini whether the inspected evidence answers the question."""
    if not evidence:
        return {
            "sufficient": False,
            "next_node_ids": [],
            "reasoning": "No readable page evidence was retrieved.",
        }

    evidence_for_prompt = [
        {
            "source_id": f"S{index + 1}",
            "filename": item["filename"],
            "page_number": item["page_number"],
            "section_title": item["section_title"],
            "text": item["text"][:3500],
        }
        for index, item in enumerate(
            evidence[-MAX_PAGES_IN_CONTEXT:]
        )
    ]

    remaining = [
        node
        for node in outline
        if node["node_id"] not in visited_ids
    ]

    prompt = f"""
You evaluate evidence for a document question-answering system.

QUESTION:
{question}

INSPECTED PAGE EVIDENCE:
{json.dumps(evidence_for_prompt, ensure_ascii=False)}

UNINSPECTED TREE SECTIONS:
{json.dumps(remaining, ensure_ascii=False)}

Return valid JSON with exactly these fields:
{{
  "sufficient": true,
  "next_node_ids": [],
  "reasoning": "Brief evidence assessment"
}}

Rules:
- Sufficient means the inspected evidence supports a complete,
  accurate answer to the actual question.
- Do not assume missing facts.
- For questions requiring multiple facts or comparisons, check
  that all necessary facts are supported.
- If insufficient, select up to {MAX_SECTIONS_PER_ROUND}
  useful uninspected section IDs.
- Choose only IDs from UNINSPECTED TREE SECTIONS.
- If sufficient, return an empty next_node_ids list.
- If no useful sections remain, return sufficient=false
  and an empty next_node_ids list.
- Return JSON only, without Markdown fences.
"""

    response = generate_content_with_retry(
        prompt,
        config={"response_mime_type": "application/json"},
    )

    result = _parse_json_response(response)

    remaining_ids = {
        node["node_id"]
        for node in remaining
    }

    next_ids = result.get("next_node_ids", [])

    if not isinstance(next_ids, list):
        next_ids = []

    next_ids = list(dict.fromkeys(
        node_id
        for node_id in next_ids
        if isinstance(node_id, str)
        and node_id in remaining_ids
    ))[:MAX_SECTIONS_PER_ROUND]

    return {
        "sufficient": result.get("sufficient") is True,
        "next_node_ids": next_ids,
        "reasoning": str(
            result.get("reasoning") or ""
        ),
    }


def _group_pages(pages: list[dict]) -> list[dict]:
    """Group retrieved pages by their associated section."""
    groups = {}

    for page in pages:
        key = (
            str(page["document_id"]),
            str(page["section_id"]),
        )

        if key not in groups:
            groups[key] = {
                "document_id": page["document_id"],
                "filename": page["filename"],
                "section_id": page["section_id"],
                "section_title": page["section_title"],
                "ancestor_titles": page["ancestor_titles"],
                "start_page": page["start_page"],
                "end_page": page["end_page"],
                "relevance_score": 1,
                "pages": [],
            }

        groups[key]["pages"].append({
            "document_id": page["document_id"],
            "filename": page["filename"],
            "page_number": page["page_number"],
            "text": page["text"],

            "blocks": page.get("blocks", []),

            "words": page.get("words", []),

            "page_width": page.get("page_width"),
            "page_height": page.get("page_height"),
            "page_rotation": page.get("page_rotation", 0),
            "coordinate_system": page.get(
                "coordinate_system",
                "pymupdf_unrotated",
            ),

            "relevance_score": 1,
        })

    return list(groups.values())


def retrieve_relevant_pages(
    question: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> dict:
    """
    Retrieve evidence through iterative LLM-guided tree navigation.

    top_k is retained for API compatibility and controls the requested
    page budget. The actual returned page count is capped for safety.
    Retrieval is based on tree navigation, not vector similarity.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be an integer of at least 1"
        )

    query_analysis = understand_query(question)

    if not isinstance(query_analysis, dict):
        raise RuntimeError(
            "Query analysis returned an invalid structure"
        )

    tree_filter = (
        {"document_id": document_id}
        if document_id
        else {}
    )

    trees = list(
        page_index_trees_collection.find(tree_filter)
    )

    if not trees:
        return {
            "question": question,
            "query_analysis": query_analysis,
            "sections_found": 0,
            "pages_found": 0,
            "results": [],
            "retrieval_trace": [],
        }

    all_pages = {}
    retrieval_trace = []
    selected_section_ids = set()

    for tree in trees:
        current_document_id = tree.get("document_id")

        if current_document_id is None:
            continue

        filename = tree.get("filename") or "Unknown document"
        sections = _collect_sections(tree)

        if not sections:
            continue

        outline = _make_tree_outline(sections)
        sections_by_id = {
            section["node_id"]: section
            for section in sections
        }

        visited_ids = set()
        queued_ids = []
        evidence_for_document = []
        round_number = 0

        while round_number < MAX_SEARCH_ROUNDS:
            round_number += 1

            if queued_ids:
                chosen_ids = queued_ids
                queued_ids = []
            else:
                chosen_ids = _select_sections(
                    question=question,
                    query_analysis=query_analysis,
                    outline=outline,
                    visited_ids=visited_ids,
                    previous_evidence=evidence_for_document,
                )

            chosen_ids = list(dict.fromkeys(
                node_id
                for node_id in chosen_ids
                if isinstance(node_id, str)
                and node_id in sections_by_id
                and node_id not in visited_ids
            ))[:MAX_SECTIONS_PER_ROUND]

            if not chosen_ids:
                break

            chosen_sections = [
                sections_by_id[node_id]
                for node_id in chosen_ids
            ]

            visited_ids.update(chosen_ids)
            selected_section_ids.update(
                (str(current_document_id), node_id)
                for node_id in chosen_ids
            )

            new_pages = _read_section_pages(
                document_id=current_document_id,
                filename=filename,
                selected_sections=chosen_sections,
            )

            for page in new_pages:
                key = (
                    str(page["document_id"]),
                    page["page_number"],
                )

                existing = all_pages.get(key)

                if (
                    existing is None
                    or len(page["ancestor_titles"])
                    > len(existing["ancestor_titles"])
                ):
                    all_pages[key] = page

                evidence_for_document.append(page)

            evidence_by_page = {}

            for page in evidence_for_document:
                key = (
                    str(page["document_id"]),
                    page["page_number"],
                )
                evidence_by_page[key] = page

            evidence_for_document = list(
                evidence_by_page.values()
            )

            assessment = _assess_evidence(
                question=question,
                evidence=evidence_for_document,
                visited_ids=visited_ids,
                outline=outline,
            )

            retrieval_trace.append({
                "document_id": current_document_id,
                "round": round_number,
                "selected_node_ids": chosen_ids,
                "pages_read": [
                    page["page_number"]
                    for page in new_pages
                ],
                "sufficient": assessment["sufficient"],
                "reasoning": assessment["reasoning"],
            })

            if assessment["sufficient"]:
                break

            queued_ids = assessment["next_node_ids"]

            if not queued_ids:
                break

    page_budget = min(
        max(top_k, 5),
        MAX_PAGES_IN_CONTEXT,
    )

    selected_pages = sorted(
        all_pages.values(),
        key=lambda page: (
            str(page["document_id"]),
            page["page_number"],
        ),
    )[:page_budget]

    return {
        "question": question,
        "query_analysis": query_analysis,
        "sections_found": len({
            (
                str(page["document_id"]),
                str(page["section_id"]),
            )
            for page in selected_pages
        }),
        "pages_found": len(selected_pages),
        "results": _group_pages(selected_pages),
        "retrieval_trace": retrieval_trace,
    }
