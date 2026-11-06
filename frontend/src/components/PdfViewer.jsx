
import { useEffect, useMemo, useRef, useState } from "react";

import {
  ChevronLeft,
  ChevronRight,
  Minus,
  Plus,
} from "lucide-react";

import { Document, Page, pdfjs } from "react-pdf";

import { getDocumentPdfUrl } from "../services/api";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

function getRectangles(citation) {
  if (!citation) return [];

  const evidence = Array.isArray(citation.evidence)
    ? citation.evidence
    : [];

  const evidenceRectangles = evidence.flatMap((item) =>
    (item.highlight_rectangles || []).map((rect) => ({
      rect,
      coordinateSystem:
        item.coordinate_system ||
        citation.coordinate_system ||
        "pymupdf_unrotated",
    }))
  );

  if (evidenceRectangles.length > 0) {
    return evidenceRectangles;
  }

  return (citation.highlight_rectangles || []).map((rect) => ({
    rect,
    coordinateSystem:
      citation.coordinate_system || "pymupdf_unrotated",
  }));
}

function normalizeRect(rect) {
  if (Array.isArray(rect) && rect.length >= 4) {
    return {
      x0: Number(rect[0]),
      y0: Number(rect[1]),
      x1: Number(rect[2]),
      y1: Number(rect[3]),
    };
  }

  if (rect && typeof rect === "object") {
    return {
      x0: Number(rect.x0),
      y0: Number(rect.y0),
      x1: Number(rect.x1),
      y1: Number(rect.y1),
    };
  }

  return null;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}


function transformRectangle(
  rect,
  pageWidth,
  pageHeight,
  rotation
) {
  const normalized = normalizeRect(rect);

  if (
    !normalized ||
    !Number.isFinite(normalized.x0) ||
    !Number.isFinite(normalized.y0) ||
    !Number.isFinite(normalized.x1) ||
    !Number.isFinite(normalized.y1)
  ) {
    return null;
  }

  const x0 = Math.min(normalized.x0, normalized.x1);
  const y0 = Math.min(normalized.y0, normalized.y1);
  const x1 = Math.max(normalized.x0, normalized.x1);
  const y1 = Math.max(normalized.y0, normalized.y1);

  let left;
  let top;
  let width;
  let height;
  let displayWidth;
  let displayHeight;

  const angle = ((rotation % 360) + 360) % 360;

  if (angle === 90) {
    left = pageHeight - y1;
    top = x0;
    width = y1 - y0;
    height = x1 - x0;

    displayWidth = pageHeight;
    displayHeight = pageWidth;
  } else if (angle === 180) {
    left = pageWidth - x1;
    top = pageHeight - y1;
    width = x1 - x0;
    height = y1 - y0;

    displayWidth = pageWidth;
    displayHeight = pageHeight;
  } else if (angle === 270) {
    left = y0;
    top = pageWidth - x1;
    width = y1 - y0;
    height = x1 - x0;

    displayWidth = pageHeight;
    displayHeight = pageWidth;
  } else {
    left = x0;
    top = y0;
    width = x1 - x0;
    height = y1 - y0;

    displayWidth = pageWidth;
    displayHeight = pageHeight;
  }

  if (!displayWidth || !displayHeight) {
    return null;
  }

  const leftPercent = clamp(
    (left / displayWidth) * 100,
    0,
    100
  );

  const topPercent = clamp(
    (top / displayHeight) * 100,
    0,
    100
  );

  const rightPercent = clamp(
    ((left + width) / displayWidth) * 100,
    0,
    100
  );

  const bottomPercent = clamp(
    ((top + height) / displayHeight) * 100,
    0,
    100
  );

  return {
    left: `${leftPercent}%`,
    top: `${topPercent}%`,
    width: `${Math.max(0, rightPercent - leftPercent)}%`,
    height: `${Math.max(0, bottomPercent - topPercent)}%`,
  };
}

export default function PdfViewer({
  documentId,
  initialPage = 1,
  citation = null,
}) {
  const [pdfError, setPdfError] = useState("");
  const [numPages, setNumPages] = useState(0);

  const [pageNumber, setPageNumber] = useState(
    Math.max(1, Number(initialPage) || 1)
  );

  const [pageWidth, setPageWidth] = useState(650);
  const [zoom, setZoom] = useState(1);
  const [pageGeometry, setPageGeometry] = useState(null);

  const documentAreaRef = useRef(null);


  const pdfUrl = useMemo(() => {
    if (!documentId) return "";

    try {
      return getDocumentPdfUrl(documentId);
    } catch (error) {
      return "";
    }
  }, [documentId]);

  useEffect(() => {
    setPdfError("");
    setNumPages(0);
    setPageGeometry(null);
    setPageNumber(
      Math.max(1, Number(initialPage) || 1)
    );
  }, [documentId, initialPage]);

  useEffect(() => {
    const nextPage = Math.max(
      1,
      Number(initialPage) || 1
    );

    setPageNumber(
      numPages > 0
        ? clamp(nextPage, 1, numPages)
        : nextPage
    );

    setPageGeometry(null);
  }, [initialPage, citation, numPages]);

  useEffect(() => {
    const element = documentAreaRef.current;

    if (!element) return undefined;

    const updateWidth = () => {
      const availableWidth = element.clientWidth - 40;

      setPageWidth(
        Math.max(220, Math.min(900, availableWidth))
      );
    };

    updateWidth();

    const observer = new ResizeObserver(updateWidth);

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const highlightRectangles = useMemo(
    () => getRectangles(citation),
    [citation]
  );

  const citationPage = Number(
    citation?.page_number || initialPage
  );

  const showHighlights =
    citationPage === pageNumber;

  function handleDocumentLoad({ numPages: totalPages }) {
    setNumPages(totalPages);
    setPdfError("");

    setPageNumber((current) =>
      clamp(current, 1, totalPages)
    );
  }

  function handlePageLoad(pdfPage) {
    const viewport = pdfPage.getViewport({
      scale: 1,
      rotation: 0,
    });

    const citationRotation = Number(
      citation?.page_rotation
    );

    const rotation =
      citation?.page_rotation != null &&
      Number.isFinite(citationRotation)
        ? citationRotation
        : pdfPage.rotate || 0;

    setPageGeometry({
      width: viewport.width,
      height: viewport.height,
      rotation,
    });
  }

  function handlePdfError(error) {
    console.error("PDF viewer error:", error);

    setPdfError(
      error?.message ||
        "Unable to load or render this PDF."
    );
  }

  function changePage(nextPage) {
    setPageNumber(
      clamp(nextPage, 1, Math.max(1, numPages))
    );

    setPageGeometry(null);
  }

  function changeZoom(amount) {
    setZoom((current) =>
      clamp(
        Number((current + amount).toFixed(2)),
        0.6,
        1.8
      )
    );
  }

  return (
    <div className="pdf-viewer-container">
      {/* PDF toolbar */}
      <div className="pdf-viewer-toolbar">
        <div className="pdf-page-controls">
          <button
            type="button"
            onClick={() => changePage(pageNumber - 1)}
            disabled={!pdfUrl || pageNumber <= 1}
            title="Previous page"
          >
            <ChevronLeft size={17} />
          </button>

          <span className="pdf-page-indicator">
            Page {pageNumber} of {numPages || "â€”"}
          </span>

          <button
            type="button"
            onClick={() => changePage(pageNumber + 1)}
            disabled={
              !pdfUrl ||
              !numPages ||
              pageNumber >= numPages
            }
            title="Next page"
          >
            <ChevronRight size={17} />
          </button>
        </div>

        <div className="pdf-page-controls">
          <button
            type="button"
            onClick={() => changeZoom(-0.1)}
            disabled={zoom <= 0.6}
            title="Zoom out"
          >
            <Minus size={15} />
          </button>

          <span className="pdf-page-indicator">
            {Math.round(zoom * 100)}%
          </span>

          <button
            type="button"
            onClick={() => changeZoom(0.1)}
            disabled={zoom >= 1.8}
            title="Zoom in"
          >
            <Plus size={15} />
          </button>
        </div>
      </div>

      {/* PDF page area */}
      <div
        className="pdf-document-area"
        ref={documentAreaRef}
      >
        {!documentId && (
          <div className="pdf-error">
            Document ID is missing.
          </div>
        )}

        {documentId && !pdfError && (
          <Document
            key={documentId}
            file={pdfUrl}
            onLoadSuccess={handleDocumentLoad}
            onLoadError={handlePdfError}
            loading={
              <div className="pdf-loading">
                Loading document...
              </div>
            }
            error={
              <div className="pdf-error">
                Unable to load this PDF.
              </div>
            }
          >
            <div className="pdf-page-wrapper">
              <Page
                pageNumber={pageNumber}
                width={pageWidth * zoom}
                onLoadSuccess={handlePageLoad}
                onLoadError={handlePdfError}
                renderTextLayer={false}
                renderAnnotationLayer={false}
                loading={
                  <div className="pdf-loading">
                    Rendering page...
                  </div>
                }
                error={
                  <div className="pdf-error">
                    Unable to render this page.
                  </div>
                }
              />

              {/* Citation highlights */}
              {showHighlights &&
                pageGeometry &&
                highlightRectangles.length > 0 && (
                  <div className="pdf-highlight-layer">
                    {highlightRectangles.map(
                      ({ rect }, index) => {
                        const position =
                          transformRectangle(
                            rect,
                            pageGeometry.width,
                            pageGeometry.height,
                            pageGeometry.rotation
                          );

                        if (!position) return null;

                        return (
                          <div
                            key={`highlight-${index}`}
                            className="pdf-highlight active"
                            style={position}
                          />
                        );
                      }
                    )}
                  </div>
                )}
            </div>
          </Document>
        )}

        {pdfError && (
          <div className="pdf-error">
            {pdfError}
          </div>
        )}
      </div>
    </div>
  );
}
