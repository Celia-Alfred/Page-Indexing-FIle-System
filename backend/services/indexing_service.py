
from uuid import uuid4

from database import (
    documents_collection,
    pages_collection,
    page_index_trees_collection,
)
from services.pdf_service import extract_pages
from services.tree_service import build_page_index_tree

def index_pdf(filename: str, pdf_bytes: bytes) -> dict:
    document_id = str(uuid4())
    extracted_pages = extract_pages(pdf_bytes)

    document_record = {
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(extracted_pages),
    }

    page_records = [
        {
            **page,
            "document_id": document_id,
            "filename": filename,
        }
        for page in extracted_pages
    ]

    page_index_tree = build_page_index_tree(
        document_id=document_id,
        filename=filename,
        pages=extracted_pages,
    )

    try:
        if page_records:
            pages_collection.insert_many(page_records)

        documents_collection.insert_one(document_record)
        page_index_trees_collection.insert_one(page_index_tree)

    except Exception:
        try:
            pages_collection.delete_many({"document_id": document_id})
            documents_collection.delete_one({"document_id": document_id})
            page_index_trees_collection.delete_one(
                {"document_id": document_id}
            )
        except Exception:
            pass

        raise

    return {
        **document_record,
        "pages_stored": len(page_records),
    }
