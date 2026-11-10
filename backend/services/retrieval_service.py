
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from database import (
    page_index_trees_collection,
    pages_collection,
)
from services.llm_service import (
    understand_query,
    generate_content_with_retry,
)


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LANGFUSE_ENABLED = (
    os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
)

LANGFUSE_LOG_CONTENT = (
    os.getenv("LANGFUSE_LOG_CONTENT", "false").lower() == "true"
)

logger = logging.getLogger(__name__)



langfuse = None

if LANGFUSE_ENABLED:
    try:
        if (
            os.getenv("LANGFUSE_PUBLIC_KEY")
            and os.getenv("LANGFUSE_SECRET_KEY")
        ):
            from langfuse import get_client

            langfuse = get_client()

            logger.info(
                "Langfuse retrieval tracing initialized successfully"
            )
        else:
            logger.warning(
                "Langfuse is enabled but API keys are missing"
            )

    except Exception:
        logger.exception(
            "Failed to initialize Langfuse. "
            "Retrieval will continue without tracing."
        )


def _start_span(
    name: str,
    metadata: dict | None = None,
):
    """
    Start a Langfuse span.

    If Langfuse is disabled or unavailable, return a
    no-op context manager so retrieval continues normally.
    """
    if langfuse is None:
        return nullcontext(None)

    try:
        return langfuse.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata or {},
        )

    except Exception:
        logger.exception(
            "Could not start Langfuse span: %s",
            name,
        )
        return nullcontext(None)


def _update_span(
    observation,
    **kwargs,
):
    """Safely update a Langfuse observation."""
    if observation is None:
        return

    try:
        observation.update(**kwargs)

    except Exception:
        logger.exception(
            "Could not update Langfuse observation"
        )



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




def _understand_query_with_trace(
    question: str,
) -> dict:
    """
    Understand the user's question and record the
    query analysis operation in Langfuse.
    """

    with _start_span(
        "pageindex-query-understanding",
        metadata={
            "operation": "query_understanding",
            "provider": "Google Gemini",
        },
    ) as observation:

        _update_span(
            observation,
            input={
                "question": question,
            },
        )

        query_analysis = understand_query(question)

        if not isinstance(query_analysis, dict):
            raise RuntimeError(
                "Query analysis returned an invalid structure"
            )

        _update_span(
            observation,
            output={
                "query_analysis": query_analysis,
                "topic": query_analysis.get("topic"),
                "keywords": query_analysis.get("keywords", []),
                "information_needed": query_analysis.get(
                    "information_needed"
                ),
            },
        )

        return query_analysis



def _valid_page_range(
    node: dict,
) -> tuple[int, int] | None:
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



def _collect_sections(
    tree: dict,
) -> list[dict]:
    """Flatten the tree while preserving its hierarchy."""

    sections = []

    def visit(
        nodes,
        ancestors=None,
    ):
        if ancestors is None:
            ancestors = []

        if not isinstance(nodes, list):
            return

        for node in nodes:

            if not isinstance(node, dict):
                continue

            next_ancestors = ancestors

            if node.get("node_type") == "section":

                title = str(
                    node.get("title") or ""
                ).strip()

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




def _make_tree_outline(
    sections: list[dict],
) -> list[dict]:
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

    with _start_span(
        "pageindex-section-selection",
        metadata={
            "available_sections": len(remaining),
            "previous_evidence_pages": len(previous_evidence),
            "selection_limit": MAX_SECTIONS_PER_ROUND,
        },
    ) as observation:

        if LANGFUSE_LOG_CONTENT:

            _update_span(
                observation,
                input={
                    "question": question,
                    "query_analysis": query_analysis,
                    "available_sections": remaining,
                    "previous_evidence": evidence_summary,
                },
            )

        response = generate_content_with_retry(
            prompt,
            config={
                "response_mime_type": "application/json"
            },
        )

        result = _parse_json_response(response)

        selected = result.get(
            "selected_node_ids",
            [],
        )

        if not isinstance(selected, list):
            selected = []

        valid_ids = {
            node["node_id"]
            for node in remaining
        }

        valid_selected = list(dict.fromkeys(
            node_id
            for node_id in selected
            if isinstance(node_id, str)
            and node_id in valid_ids
        ))[:MAX_SECTIONS_PER_ROUND]

        _update_span(
            observation,
            output={
                "selected_node_ids": valid_selected,
                "reasoning": result.get("reasoning", ""),
            },
        )

        return valid_selected



def _read_section_pages_impl(
    document_id: Any,
    filename: str,
    selected_sections: list[dict],
) -> list[dict]:
    """
    Read original page text and preserve all OCR/native
    extraction and coordinate metadata.
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
            "lines": 1,
            "words": 1,
            "extraction_source": 1,
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

        text = (
            page.get("text") or ""
        ).strip()

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
            "lines": page.get("lines", []),
            "words": page.get("words", []),

            "extraction_source": page.get(
                "extraction_source",
                "native",
            ),

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


def _read_section_pages(
    document_id: Any,
    filename: str,
    selected_sections: list[dict],
) -> list[dict]:
    """Retrieve pages and record the operation in Langfuse."""

    if not selected_sections:
        return []

    with _start_span(
        "pageindex-page-retrieval",
        metadata={
            "document_id": str(document_id),
            "filename": filename,
            "selected_sections": [
                section["node_id"]
                for section in selected_sections
            ],
        },
    ) as observation:

        pages = _read_section_pages_impl(
            document_id=document_id,
            filename=filename,
            selected_sections=selected_sections,
        )

        page_details = [
            {
                "page_number": page["page_number"],
                "section_id": page["section_id"],
                "section_title": page["section_title"],
                "extraction_source": page["extraction_source"],
            }
            for page in pages
        ]

        output = {
            "pages_retrieved": len(pages),
            "pages": page_details,
        }

        if LANGFUSE_LOG_CONTENT:

            output["retrieved_evidence"] = [
                {
                    "page_number": page["page_number"],
                    "text": page["text"],
                }
                for page in pages
            ]

        _update_span(
            observation,
            output=output,
        )

        return pages


def _assess_evidence_impl(
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
        config={
            "response_mime_type": "application/json"
        },
    )

    result = _parse_json_response(response)

    remaining_ids = {
        node["node_id"]
        for node in remaining
    }

    next_ids = result.get(
        "next_node_ids",
        [],
    )

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


def _assess_evidence(
    question: str,
    evidence: list[dict],
    visited_ids: set[str],
    outline: list[dict],
) -> dict:
    """Assess retrieved evidence and record the decision in Langfuse."""

    with _start_span(
        "pageindex-evidence-assessment",
        metadata={
            "evidence_pages": len(evidence),
            "visited_sections": len(visited_ids),
            "remaining_sections": sum(
                1
                for node in outline
                if node["node_id"] not in visited_ids
            ),
        },
    ) as observation:

        if LANGFUSE_LOG_CONTENT:

            _update_span(
                observation,
                input={
                    "question": question,
                    "evidence": [
                        {
                            "page_number": item["page_number"],
                            "section_title": item["section_title"],
                            "text": item["text"],
                        }
                        for item in evidence
                    ],
                },
            )

        result = _assess_evidence_impl(
            question=question,
            evidence=evidence,
            visited_ids=visited_ids,
            outline=outline,
        )

        _update_span(
            observation,
            output=result,
        )

        return result



def _group_pages(
    pages: list[dict],
) -> list[dict]:
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
            "lines": page.get("lines", []),
            "words": page.get("words", []),

            "extraction_source": page.get(
                "extraction_source",
                "native",
            ),

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



def _retrieve_relevant_pages_impl(
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


    query_analysis = _understand_query_with_trace(
        question
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

        filename = (
            tree.get("filename")
            or "Unknown document"
        )

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
                if (
                    isinstance(node_id, str)
                    and node_id in sections_by_id
                    and node_id not in visited_ids
                )
            ))[:MAX_SECTIONS_PER_ROUND]

            if not chosen_ids:
                break

            chosen_sections = [
                sections_by_id[node_id]
                for node_id in chosen_ids
            ]

            visited_ids.update(chosen_ids)

            selected_section_ids.update(
                (
                    str(current_document_id),
                    node_id,
                )
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

            round_details = {
                "document_id": str(current_document_id),
                "round": round_number,
                "selected_node_ids": chosen_ids,
                "pages_read": [
                    page["page_number"]
                    for page in new_pages
                ],
                "sufficient": assessment["sufficient"],
                "reasoning": assessment["reasoning"],
                "next_node_ids": assessment["next_node_ids"],
            }

            retrieval_trace.append(round_details)


            with _start_span(
                f"pageindex-search-round-{round_number}",
                metadata={
                    "document_id": str(current_document_id),
                    "round": round_number,
                    "selected_node_ids": chosen_ids,
                    "pages_read": [
                        page["page_number"]
                        for page in new_pages
                    ],
                    "evidence_sufficient": assessment["sufficient"],
                },
            ) as observation:

                _update_span(
                    observation,
                    output={
                        "selected_sections": chosen_ids,
                        "pages_retrieved": [
                            page["page_number"]
                            for page in new_pages
                        ],
                        "evidence_sufficient": assessment["sufficient"],
                        "next_sections": assessment["next_node_ids"],
                        "reasoning": assessment["reasoning"],
                    },
                )


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

    grouped_results = _group_pages(
        selected_pages
    )

    sections_found = len({
        (
            str(page["document_id"]),
            str(page["section_id"]),
        )
        for page in selected_pages
    })


    return {
        "question": question,
        "query_analysis": query_analysis,
        "sections_found": sections_found,
        "pages_found": len(selected_pages),
        "results": grouped_results,
        "retrieval_trace": retrieval_trace,
    }


def retrieve_relevant_pages(
    question: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> dict:
    """
    Public retrieval function.

    Creates the main PageIndex retrieval trace.
    Child operations are recorded as nested observations.
    """

    with _start_span(
        "pageindex-retrieval",
        metadata={
            "document_id": (
                str(document_id)
                if document_id
                else "all"
            ),
            "top_k": top_k,
            "retrieval_method": (
                "vectorless_hierarchical_tree_search"
            ),
            "max_search_rounds": MAX_SEARCH_ROUNDS,
        },
    ) as observation:

        _update_span(
            observation,
            input={
                "question": question,
                "document_id": (
                    str(document_id)
                    if document_id
                    else None
                ),
                "top_k": top_k,
            },
        )

        result = _retrieve_relevant_pages_impl(
            question=question,
            document_id=document_id,
            top_k=top_k,
        )


        retrieved_pages = []

        for group in result["results"]:

            for page in group["pages"]:

                retrieved_pages.append({
                    "filename": group["filename"],
                    "section_id": group["section_id"],
                    "section_title": group["section_title"],
                    "page_number": page["page_number"],
                    "extraction_source": page.get(
                        "extraction_source",
                        "native",
                    ),
                })

        trace_output = {
            "query_analysis": result["query_analysis"],
            "sections_found": result["sections_found"],
            "pages_found": result["pages_found"],
            "retrieved_pages": retrieved_pages,
            "retrieval_rounds": result["retrieval_trace"],
        }

        if LANGFUSE_LOG_CONTENT:

            trace_output["retrieved_results"] = result["results"]

        _update_span(
            observation,
            output=trace_output,
        )

        return result
