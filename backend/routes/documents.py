
import logging
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from database import documents_collection
from services.indexing_service import index_pdf
from services.retrieval_service import retrieve_relevant_pages
from services.answer_service import generate_answer
from utils.file_validation import validate_pdf

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await validate_pdf(file)
        filename = file.filename or "document.pdf"

        # index_pdf must also persist the original PDF bytes.
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
            "PDF indexed: filename=%s, document_id=%s",
            response["filename"],
            response["document_id"],
        )

        return jsonable_encoder(response)

    except HTTPException:
        raise

    except PyMongoError:
        logger.exception("MongoDB error during PDF upload")
        raise HTTPException(
            status_code=500,
            detail="Failed to store PDF in MongoDB.",
        )

    except Exception:
        logger.exception("Unexpected error during PDF upload")
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check the backend terminal.",
        )


@router.post("/query")
def query_documents(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        retrieval_result = retrieve_relevant_pages(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )

        answer_result = generate_answer(
            question=request.question,
            retrieved_sections=retrieval_result.get("results", []),
        )

        sections = retrieval_result.get("results", [])

        # Citation objects retain the metadata required by the UI.
        citations = answer_result.get("citations", [])

        response = {
            "question": request.question,
            "document_id": request.document_id,
            "query_analysis": retrieval_result.get("query_analysis"),
            "sections_found": retrieval_result.get("sections_found", 0),
            "answer": answer_result["answer"],
            "citations": citations,
            "sources": [
                {
                    "filename": section.get("filename", "Unknown document"),
                    "document_id": section.get("document_id"),
                    "section": section.get("section_title", "Unknown section"),
                    "start_page": section.get("start_page"),
                    "end_page": section.get("end_page"),
                    "relevance_score": section.get("relevance_score"),
                }
                for section in sections
            ],
        }

        return jsonable_encoder(response)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except PyMongoError:
        logger.exception("MongoDB error during document retrieval")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents from MongoDB.",
        )

    except Exception:
        logger.exception("Unexpected error during document query")
        raise HTTPException(
            status_code=500,
            detail="Document query failed. Check the backend terminal.",
        )


@router.get("/{document_id}/pdf")
def get_document_pdf(
    document_id: str,
    page: int | None = Query(default=None, ge=1),
):
    """
    Serve the original PDF for inline browser viewing.

    The optional page parameter is intended for citation navigation.
    The browser viewer may need a frontend PDF viewer to open that page
    automatically.
    """
    try:
        document = documents_collection.find_one(
            {"document_id": document_id},
            {"filename": 1, "pdf_bytes": 1},
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

        filename = document.get("filename", "document.pdf")
        safe_filename = quote(filename)

        headers = {
            "Content-Disposition": (
                f"inline; filename*=UTF-8''{safe_filename}"
            ),
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-cache",
        }

        # The page value is accepted for citation-link consistency.
        # Actual page navigation is handled by the frontend PDF viewer.
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers=headers,
        )

    except HTTPException:
        raise

    except PyMongoError:
        logger.exception("MongoDB error while retrieving original PDF")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve the original PDF.",
        )

    except Exception:
        logger.exception("Unexpected error while retrieving original PDF")
        raise HTTPException(
            status_code=500,
            detail="Unable to open the original PDF.",
        )
