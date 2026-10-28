
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

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
    top_k: int = Field(default=3, ge=1)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await validate_pdf(file)

        result = index_pdf(file.filename, pdf_bytes)

        response = {
            "message": "PDF uploaded and indexed successfully",
            "document_id": str(result["document_id"]),
            "filename": str(result["filename"]),
            "total_pages": int(result["total_pages"]),
            "pages_stored": int(result["pages_stored"]),
        }

        logger.info(
            "PDF uploaded successfully: filename=%s, document_id=%s",
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
            detail="Failed to store PDF in MongoDB. Check the backend terminal.",
        )

    except Exception:
        logger.exception("Unexpected error during PDF upload")
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check the backend terminal.",
        )


@router.post("/query")
def query_documents(request: QueryRequest):
    if not request.question.strip():
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
            retrieved_sections=retrieval_result["results"],
        )

        response = {
            "question": request.question,
            "document_id": request.document_id,
            "query_analysis": retrieval_result["query_analysis"],
            "sections_found": retrieval_result["sections_found"],
            "answer": answer_result["answer"],
            "citations": answer_result["citations"],
            "sources": [
                {
                    "filename": section["filename"],
                    "document_id": section["document_id"],
                    "section": section["section_title"],
                    "start_page": section["start_page"],
                    "end_page": section["end_page"],
                    "relevance_score": section["relevance_score"],
                }
                for section in retrieval_result["results"]
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
            detail="Failed to retrieve documents from MongoDB",
        )

    except Exception:
        logger.exception("Unexpected error during document query")
        raise HTTPException(
            status_code=500,
            detail="Document query failed. Check the backend terminal.",
        )
