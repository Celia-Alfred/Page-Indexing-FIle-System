
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from pymongo.errors import PyMongoError

from database import documents_collection
from services.indexing_service import index_pdf
from services.retrieval_service import retrieve_relevant_pages
from services.answer_service import generate_answer
from utils.file_validation import validate_pdf



ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LANGFUSE_ENABLED = (
    os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
)

LANGFUSE_LOG_CONTENT = (
    os.getenv("LANGFUSE_LOG_CONTENT", "false").lower() == "true"
)

langfuse = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import get_client

        langfuse = get_client()

        logger_placeholder = logging.getLogger(__name__)
        logger_placeholder.info("Langfuse query tracing initialized.")

    except Exception:
        logging.getLogger(__name__).exception(
            "Langfuse initialization failed. "
            "Query processing will continue without route-level tracing."
        )


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ---------------------------------------------------------
# LANGFUSE HELPERS
# ---------------------------------------------------------

def start_query_observation(question: str, document_id: str | None, top_k: int):
    """
    Starts the parent observation for the complete
    document question-answering workflow.

    Retrieval and answer-generation observations created
    inside this context become child observations.
    """

    if not LANGFUSE_ENABLED or langfuse is None:
        return nullcontext(None)

    metadata = {
        "document_id": document_id,
        "top_k": top_k,
        "workflow": "page-indexing-question-answering",
    }

    observation_input = {
        "question": question,
        "document_id": document_id,
        "top_k": top_k,
    }

    if LANGFUSE_LOG_CONTENT:
        metadata["content_logging_enabled"] = True

    return langfuse.start_as_current_observation(
        as_type="span",
        name="pageindex-question-answering",
        input=observation_input,
        metadata=metadata,
    )


def update_observation(observation, **kwargs):
    """Safely updates a Langfuse observation."""

    if observation is None:
        return

    try:
        observation.update(**kwargs)

    except Exception:
        logger.exception(
            "Failed to update Langfuse observation."
        )


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None):
        if value is not None:
            value = value.strip()

            if not value:
                return None

        return value



@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its pages, build the page index tree,
    and store the document and page records in MongoDB.
    """

    try:
        pdf_bytes = await validate_pdf(file)

        filename = file.filename or "document.pdf"

        result = index_pdf(
            filename=filename,
            pdf_bytes=pdf_bytes,
        )

        response = {
            "message": "PDF uploaded and indexed successfully",
            "document_id": str(result["document_id"]),
            "filename": str(result["filename"]),
            "total_pages": int(result["total_pages"]),
            "pages_stored": int(result["pages_stored"]),
        }

        logger.info(
            "PDF indexed successfully: filename=%s, "
            "document_id=%s, pages=%s",
            response["filename"],
            response["document_id"],
            response["pages_stored"],
        )

        return jsonable_encoder(response)

    except HTTPException:
        raise

    except PyMongoError:
        logger.exception(
            "MongoDB error during PDF upload"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to store PDF in MongoDB.",
        )

    except Exception:
        logger.exception(
            "Unexpected error during PDF upload"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed. "
                "Check the backend terminal."
            ),
        )

    finally:
        await file.close()



@router.post("/query")
def query_documents(request: QueryRequest):
    """
    Retrieve relevant PDF pages and generate an answer
    with verified citations and PDF highlight coordinates.

    Supports citations from:
    - Native/digital PDF text
    - OCR-extracted scanned PDF text

    Langfuse tracks the complete workflow:
    - User query
    - Retrieval process
    - Retrieved sections
    - Answer generation
    - Citation generation
    """

    try:
        with start_query_observation(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        ) as query_observation:


            retrieval_result = retrieve_relevant_pages(
                question=request.question,
                document_id=request.document_id,
                top_k=request.top_k,
            )

            sections = retrieval_result.get("results", [])

            query_analysis = retrieval_result.get(
                "query_analysis"
            )

            retrieval_trace = retrieval_result.get(
                "retrieval_trace",
                [],
            )

            retrieval_summary = {
                "sections_found": len(sections),
                "retrieval_rounds": len(retrieval_trace),
                "query_analysis": query_analysis,
            }

            if LANGFUSE_LOG_CONTENT:
                retrieval_summary["retrieval_trace"] = retrieval_trace

                retrieval_summary["retrieved_sections"] = [
                    {
                        "filename": section.get("filename"),
                        "document_id": section.get("document_id"),
                        "section_title": section.get("section_title"),
                        "start_page": section.get("start_page"),
                        "end_page": section.get("end_page"),
                        "content": section.get("content"),
                    }
                    for section in sections
                ]


            if not sections:

                response = {
                    "question": request.question,
                    "document_id": request.document_id,
                    "query_analysis": query_analysis,
                    "sections_found": 0,
                    "answer": (
                        "I could not find relevant information "
                        "in the uploaded documents."
                    ),
                    "citations": [],
                    "sources": [],
                }

                update_observation(
                    query_observation,
                    output={
                        "status": "no_relevant_sections_found",
                        "sections_found": 0,
                        "citations_generated": 0,
                    },
                    metadata={
                        "retrieval_status": "no_results",
                    },
                )

                logger.info(
                    "No relevant document sections found: "
                    "document_id=%s",
                    request.document_id,
                )

                return jsonable_encoder(response)

            answer_result = generate_answer(
                question=request.question,
                retrieved_sections=sections,
            )

            citations = answer_result.get("citations", [])

       

            sources = []

            for section in sections:
                sources.append({
                    "filename": section.get(
                        "filename",
                        "Unknown document",
                    ),
                    "document_id": section.get("document_id"),
                    "section": section.get(
                        "section_title",
                        "Unknown section",
                    ),
                    "start_page": section.get("start_page"),
                    "end_page": section.get("end_page"),
                    "relevance_score": section.get(
                        "relevance_score"
                    ),
                })


            answer_text = answer_result.get("answer", "")

            response = {
                "question": request.question,
                "document_id": request.document_id,
                "query_analysis": query_analysis,
                "sections_found": len(sections),
                "answer": answer_text,
                "citations": citations,
                "sources": sources,
            }

       

            trace_output = {
                "status": "success",
                "sections_found": len(sections),
                "citations_generated": len(citations),
                "sources_found": len(sources),
                "retrieval_rounds": len(retrieval_trace),
            }

            if LANGFUSE_LOG_CONTENT:
                trace_output["answer"] = answer_text
                trace_output["citations"] = citations
                trace_output["sources"] = sources

            update_observation(
                query_observation,
                output=trace_output,
                metadata={
                    "retrieval_status": "completed",
                    "answer_status": "generated",
                },
            )

            logger.info(
                "Document query completed: document_id=%s, "
                "sections=%s, citations=%s",
                request.document_id,
                len(sections),
                len(citations),
            )

            return jsonable_encoder(response)

    except ValueError as exc:
        logger.warning(
            "Invalid document query: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except PyMongoError:
        logger.exception(
            "MongoDB error during document retrieval"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents from MongoDB.",
        )

    except Exception:
        logger.exception(
            "Unexpected error during document query"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document query failed. "
                "Check the backend terminal."
            ),
        )



@router.get("/{document_id}/pdf")
def get_document_pdf(
    document_id: str,
    page: int | None = Query(
        default=None,
        ge=1,
    ),
):
    """
    Retrieve and serve the original PDF.

    The frontend PDF viewer uses citation coordinates
    to navigate to pages and render highlights.
    """

    try:
        document = documents_collection.find_one(
            {"document_id": document_id},
            {
                "filename": 1,
                "pdf_bytes": 1,
            },
        )

        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        pdf_bytes = document.get("pdf_bytes")

        if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Original PDF is not stored for this document. "
                    "Re-upload it after enabling PDF storage."
                ),
            )

        filename = document.get(
            "filename",
            "document.pdf",
        )

        safe_filename = quote(
            filename,
            safe="",
        )

        headers = {
            "Content-Disposition": (
                f"inline; filename*=UTF-8''{safe_filename}"
            ),
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        }

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers=headers,
        )

    except HTTPException:
        raise

    except PyMongoError:
        logger.exception(
            "MongoDB error while retrieving original PDF"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve the original PDF.",
        )

    except Exception:
        logger.exception(
            "Unexpected error while retrieving original PDF"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to open the original PDF.",
        )
