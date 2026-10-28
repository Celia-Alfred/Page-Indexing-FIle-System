
import pymupdf

from services.pdf_service import extract_pages


def test_extract_pages():
    pdf = pymupdf.open()

    page1 = pdf.new_page()
    page1.insert_text((72, 72), "Content from page one")

    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Content from page two")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    pages = extract_pages(pdf_bytes)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Content from page one" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Content from page two" in pages[1]["text"]
