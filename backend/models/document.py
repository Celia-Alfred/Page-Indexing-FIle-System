
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    pages_stored: int
