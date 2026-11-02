
import { useState, useEffect, useRef } from "react";
import {
  MessageSquare,
  Sparkles,
  FileText,
  LoaderCircle,
  Send,
  ArrowUpRight,
  Bot,
  UserRound,
  Search,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from "lucide-react";

import { Document, Page, pdfjs } from "react-pdf";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

const API_BASE = "http://127.0.0.1:8000";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export default function ChatPage({
  documents = [],
  selectedDocument,
  setSelectedDocument,
  messages = [],
  setMessages,
  setActivePage,
  selectedDoc,
}) {
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});

  const [pdfViewer, setPdfViewer] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState(false);
  const [pdfZoom, setPdfZoom] = useState(100);
  const [pdfPageCount, setPdfPageCount] = useState(0);
  const [pdfContainerWidth, setPdfContainerWidth] = useState(700);

  const messagesEndRef = useRef(null);
  const pdfContentRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, asking]);

  useEffect(() => {
    const container = pdfContentRef.current;

    if (!container || !pdfViewer) return;

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect?.width;

      if (width) {
        setPdfContainerWidth(width);
      }
    });

    observer.observe(container);

    return () => observer.disconnect();
  }, [pdfViewer]);

  async function askQuestion(event) {
    event.preventDefault();

    const questionText = question.trim();

    if (!questionText || asking) return;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        text: questionText,
      },
    ]);

    setQuestion("");
    setAsking(true);

    try {
      const payload = {
        question: questionText,
        top_k: 3,
      };

      if (selectedDocument) {
        payload.document_id = selectedDocument;
      }

      const response = await fetch(`${API_BASE}/documents/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to answer your question."
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.answer || "I couldn't generate an answer.",
          citations: Array.isArray(data.citations)
            ? data.citations
            : [],
          sources: Array.isArray(data.sources)
            ? data.sources
            : [],
          sectionsFound: data.sections_found,
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: "error",
          text: error.message.includes("Failed to fetch")
            ? "Cannot reach the backend. Check that FastAPI is running and CORS is configured."
            : error.message,
        },
      ]);
    } finally {
      setAsking(false);
    }
  }

  function useSuggestion(text) {
    setQuestion(text);
    document.getElementById("question-input")?.focus();
  }

  // --------------------------------------------------
  // CITATION HELPERS
  // --------------------------------------------------

  function getCitationId(citation, index) {
    return String(
      citation.source_id || citation.id || `S${index + 1}`
    ).replace(/^\[|\]$/g, "");
  }

  function getPageNumber(citation) {
    const page =
      citation.page_number ??
      citation.page ??
      citation.pageNumber;

    const parsed = Number(page);

    return Number.isInteger(parsed) && parsed > 0
      ? parsed
      : null;
  }

  function getFilename(citation) {
    return (
      citation.filename ||
      citation.file_name ||
      citation.document_filename ||
      "Source document"
    );
  }

  function getDocumentId(citation) {
    return (
      citation.document_id ||
      citation.documentId ||
      citation.document?.document_id ||
      ""
    );
  }

  function getDocumentForCitation(citation) {
    const documentId = getDocumentId(citation);
    const filename = getFilename(citation);

    return documents.find((doc) => {
      const id = doc.document_id || doc.id || doc._id;

      return (
        (documentId && String(id) === String(documentId)) ||
        (!documentId && filename && doc.filename === filename)
      );
    });
  }

  // --------------------------------------------------
  // EXTRACT CITATION HIGHLIGHTS
  // --------------------------------------------------

  function getCitationHighlights(citation) {
    const evidence = Array.isArray(citation.evidence)
      ? citation.evidence
      : [];

    return evidence.flatMap((item) => {
      const rectangles = Array.isArray(item.highlight_rectangles)
        ? item.highlight_rectangles
        : [];

      return rectangles
        .filter(
          (rectangle) =>
            Array.isArray(rectangle) &&
            rectangle.length === 4 &&
            rectangle.every(
              (value) =>
                typeof value === "number" &&
                Number.isFinite(value)
            )
        )
        .map((rectangle) => ({
          rectangle,
          pageNumber: Number(
            item.page_number ??
              citation.page_number ??
              0
          ),
          pageWidth: Number(item.page_width || 0),
          pageHeight: Number(item.page_height || 0),
          pageRotation: Number(item.page_rotation || 0),
          coordinateSystem:
            item.coordinate_system || "pymupdf_unrotated",
        }));
    });
  }

  // --------------------------------------------------
  // OPEN CITATION
  // --------------------------------------------------

  function openCitation(citation, citationId) {
    console.log("========== CITATION CHECK ==========");
    console.log("Citation clicked:", citationId);
    console.log("Citation object:", citation);
    console.log(
      "FULL CITATION EVIDENCE:",
      JSON.stringify(citation.evidence, null, 2)
    );

    if (!citation) {
      console.error("Citation object is missing");
      return;
    }

    const matchedDocument = getDocumentForCitation(citation);
    const highlights = getCitationHighlights(citation);

    console.log("Document:", matchedDocument);
    console.log("Evidence:", citation.evidence);

    console.log(
      "Highlight rectangles:",
      highlights.map((item) => ({
        page: item.pageNumber,
        rectangle: item.rectangle,
      }))
    );

    console.log("====================================");

    const documentId =
      getDocumentId(citation) ||
      matchedDocument?.document_id ||
      matchedDocument?.id ||
      matchedDocument?._id;

    if (!documentId) {
      console.error(
        "Document ID missing from citation:",
        citation
      );
      return;
    }

    const normalizedDocumentId = String(documentId);

    const isSameDocument =
      pdfViewer?.documentId === normalizedDocumentId;

    if (matchedDocument && setSelectedDocument) {
      setSelectedDocument(
        matchedDocument.document_id ||
          matchedDocument.id ||
          matchedDocument._id
      );
    }

    const firstEvidencePage = highlights.find(
      (item) => item.pageNumber > 0
    )?.pageNumber;

    const pageNumber =
      firstEvidencePage || getPageNumber(citation) || 1;

    const pdfUrl =
      `${API_BASE}/documents/${encodeURIComponent(normalizedDocumentId)}/pdf`;

    console.log("Selected PDF page:", pageNumber);
    console.log("Same document already open:", isSameDocument);

    setPdfViewer({
      documentId: normalizedDocumentId,
      filename: getFilename(citation),
      pageNumber,
      url: pdfUrl,
      citationId,
      highlights,
    });

    setPdfLoading(!isSameDocument);
    setPdfError(false);
    setPdfZoom(100);

    if (!isSameDocument) {
      setPdfPageCount(0);
    }
  }

  // --------------------------------------------------
  // CLOSE PDF VIEWER
  // --------------------------------------------------

  function closePdfViewer() {
    setPdfViewer(null);
    setPdfLoading(false);
    setPdfError(false);
    setPdfZoom(100);
    setPdfPageCount(0);
  }

  // --------------------------------------------------
  // CHANGE PDF PAGE
  // --------------------------------------------------

  function changePdfPage(page) {
    if (!pdfViewer || page < 1) return;

    if (pdfPageCount && page > pdfPageCount) return;

    setPdfLoading(true);
    setPdfError(false);

    setPdfViewer((previous) => ({
      ...previous,
      pageNumber: page,
    }));
  }

  // --------------------------------------------------
  // HIGHLIGHT RENDERING
  // --------------------------------------------------

  function renderCitationHighlights() {
    if (!pdfViewer?.highlights?.length) return null;

    const currentPageHighlights = pdfViewer.highlights.filter(
      (item) => item.pageNumber === pdfViewer.pageNumber
    );

    if (!currentPageHighlights.length) {
      console.warn(
        "No highlights found for PDF page:",
        pdfViewer.pageNumber
      );
      return null;
    }

    return (
      <div className="pdf-highlight-layer">
        {currentPageHighlights.map((item, index) => {
          const [x0, y0, x1, y1] = item.rectangle;

          if (
            !item.pageWidth ||
            !item.pageHeight ||
            item.pageRotation !== 0 ||
            item.coordinateSystem !== "pymupdf_unrotated" ||
            ![x0, y0, x1, y1].every(Number.isFinite) ||
            x1 <= x0 ||
            y1 <= y0
          ) {
            return null;
          }

          const left = (x0 / item.pageWidth) * 100;
          const top = (y0 / item.pageHeight) * 100;
          const width = ((x1 - x0) / item.pageWidth) * 100;
          const height = ((y1 - y0) / item.pageHeight) * 100;

          return (
            <div
              key={`highlight-${index}`}
              className="pdf-highlight-rectangle"
              style={{
                left: `${left}%`,
                top: `${top}%`,
                width: `${width}%`,
                height: `${height}%`,
              }}
            />
          );
        })}
      </div>
    );
  }

  // --------------------------------------------------
  // SOURCES
  // --------------------------------------------------

  function toggleSources(messageIndex) {
    setExpandedSources((previous) => ({
      ...previous,
      [messageIndex]: !previous[messageIndex],
    }));
  }

  // --------------------------------------------------
  // ANSWER RENDERING
  // --------------------------------------------------

  function renderAnswer(text, citations = []) {
    if (!text) return null;

    const citationMap = new Map(
      citations.map((citation, index) => [
        getCitationId(citation, index),
        citation,
      ])
    );

    const parts = text.split(/(\[S\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/^\[(S\d+)\]$/);

      if (!match) {
        return <span key={`text-${index}`}>{part}</span>;
      }

      const citation = citationMap.get(match[1]);

      if (!citation) {
        return <span key={`unknown-${index}`}>{part}</span>;
      }

      const pageNumber = getPageNumber(citation);
      const filename = getFilename(citation);

      const citationLabel =
        pageNumber != null
          ? `${filename} p.${pageNumber}`
          : filename;

      return (
        <button
          key={`citation-${index}`}
          type="button"
          className="inline-citation"
          title={`Open ${filename}, page ${pageNumber ?? "unknown"}`}
          onClick={() => openCitation(citation, match[1])}
        >
          {citationLabel}
          <ExternalLink size={11} />
        </button>
      );
    });
  }

  // --------------------------------------------------
  // PDF SIDE PANEL
  // --------------------------------------------------

  function renderPdfViewer() {
    if (!pdfViewer) return null;

    const pdfPageUrl =
      `${pdfViewer.url}#page=${pdfViewer.pageNumber}`;

    const pageWidth = Math.max(
      280,
      Math.min(pdfContainerWidth - 48, 1000) *
        (pdfZoom / 100)
    );

    return (
      <aside className="pdf-side-panel">
        <header className="pdf-viewer-header">
          <div className="pdf-viewer-title">
            <FileText size={21} />

            <div>
              <strong>{pdfViewer.filename}</strong>

              <span>
                Page {pdfViewer.pageNumber}
                {pdfPageCount > 0 ? ` of ${pdfPageCount}` : ""}
                {" · Original PDF"}
              </span>
            </div>
          </div>

          <button
            type="button"
            className="pdf-viewer-close"
            onClick={closePdfViewer}
            aria-label="Close PDF viewer"
            title="Close PDF viewer"
          >
            ×
          </button>
        </header>

        <div className="pdf-viewer-toolbar">
          <div className="pdf-page-controls">
            <button
              type="button"
              onClick={() =>
                changePdfPage(pdfViewer.pageNumber - 1)
              }
              disabled={pdfViewer.pageNumber <= 1}
              title="Previous page"
            >
              <ChevronRight
                size={17}
                style={{ transform: "rotate(180deg)" }}
              />
            </button>

            <span>
              Page {pdfViewer.pageNumber}
              {pdfPageCount > 0 ? ` / ${pdfPageCount}` : ""}
            </span>

            <button
              type="button"
              onClick={() =>
                changePdfPage(pdfViewer.pageNumber + 1)
              }
              disabled={
                pdfPageCount > 0 &&
                pdfViewer.pageNumber >= pdfPageCount
              }
              title="Next page"
            >
              <ChevronRight size={17} />
            </button>
          </div>

          <div className="pdf-zoom-controls">
            <button
              type="button"
              onClick={() =>
                setPdfZoom((previous) =>
                  Math.max(50, previous - 10)
                )
              }
              disabled={pdfZoom <= 50}
              title="Zoom out"
            >
              <ZoomOut size={17} />
            </button>

            <span>{pdfZoom}%</span>

            <button
              type="button"
              onClick={() =>
                setPdfZoom((previous) =>
                  Math.min(200, previous + 10)
                )
              }
              disabled={pdfZoom >= 200}
              title="Zoom in"
            >
              <ZoomIn size={17} />
            </button>

            <button
              type="button"
              onClick={() => window.open(pdfPageUrl, "_blank")}
              title="Open original PDF in new tab"
            >
              <Maximize2 size={16} />
            </button>
          </div>
        </div>

        <div className="pdf-viewer-content" ref={pdfContentRef}>
          {pdfLoading && !pdfError && (
            <div className="pdf-viewer-loading">
              <LoaderCircle className="spin" size={25} />
              <span>Loading original PDF...</span>
            </div>
          )}

          {pdfError ? (
            <div className="pdf-viewer-error">
              <FileText size={35} />

              <strong>Unable to display this PDF</strong>

              <p>
                Check that the backend PDF endpoint is available
                and returns the original PDF.
              </p>

              <button
                type="button"
                onClick={() => window.open(pdfPageUrl, "_blank")}
              >
                Open PDF in new tab
              </button>
            </div>
          ) : (
            <Document
              key={pdfViewer.documentId}
              file={pdfViewer.url}
              loading={null}
              onLoadSuccess={({ numPages }) => {
                console.log(
                  "PDF loaded successfully. Pages:",
                  numPages
                );

                setPdfPageCount(numPages);
                setPdfError(false);
                setPdfLoading(false);

                if (pdfViewer.pageNumber > numPages) {
                  setPdfViewer((previous) => ({
                    ...previous,
                    pageNumber: numPages,
                  }));
                }
              }}
              onLoadError={(error) => {
                console.error("PDF loading error:", error);
                setPdfLoading(false);
                setPdfError(true);
              }}
              onSourceError={(error) => {
                console.error("PDF source error:", error);
                setPdfLoading(false);
                setPdfError(true);
              }}
            >
              <div className="pdf-viewer-document">
                <div className="pdf-page-shell">
                  <Page
                    key={`${pdfViewer.documentId}-${pdfViewer.pageNumber}-${pdfZoom}`}
                    pageNumber={pdfViewer.pageNumber}
                    width={pageWidth}
                    renderTextLayer
                    renderAnnotationLayer
                    onRenderSuccess={() => {
                      console.log(
                        "PDF page rendered:",
                        pdfViewer.pageNumber
                      );
                      setPdfLoading(false);
                    }}
                    onRenderError={(error) => {
                      console.error(
                        "PDF rendering error:",
                        error
                      );
                      setPdfLoading(false);
                      setPdfError(true);
                    }}
                  />

                  {renderCitationHighlights()}
                </div>
              </div>
            </Document>
          )}
        </div>
      </aside>
    );
  }

  // --------------------------------------------------
  // MAIN UI
  // --------------------------------------------------

  return (
    <div className={`chat-page ${pdfViewer ? "pdf-open" : ""}`}>
      <div className="chat-workspace">
        <div className="chat-main-panel">
          <section className="welcome-section chat-welcome">
            <div className="welcome-copy">
              <div className="eyebrow">
                <Sparkles size={14} />
                YOUR DOCUMENTS, YOUR ANSWERS
              </div>

              <h2>
                What do you want to <span>discover?</span>
              </h2>

              <p>
                Ask questions and get AI-generated answers grounded in
                your indexed PDF content.
              </p>
            </div>
          </section>

          {documents.length === 0 && (
            <div className="chat-no-documents">
              <FileText size={19} />

              <div className="chat-no-documents-copy">
                <strong>No documents uploaded yet</strong>

                <span>
                  Upload a PDF before asking questions about your documents.
                </span>
              </div>

              <button
                type="button"
                className="chat-upload-link"
                onClick={() => setActivePage("documents")}
              >
                Upload PDF <ArrowUpRight size={15} />
              </button>
            </div>
          )}

          <section className="panel chat-panel">
            <header className="chat-panel-header">
              <div className="chat-assistant-avatar">
                <Bot size={21} />
              </div>

              <div className="chat-assistant-details">
                <h3>AI document assistant</h3>

                <span>
                  <span className="chat-online-dot" />
                  Ready to help with your documents
                </span>
              </div>

              <div className="chat-header-badge">
                <Sparkles size={13} />
                AI powered
              </div>
            </header>

            <div className="chat-scope">
              <div className="chat-scope-left">
                <Search size={15} />

                <span className="scope-label">Searching in</span>

                <strong>
                  {selectedDocument
                    ? selectedDoc?.filename || "Selected document"
                    : "All indexed documents"}
                </strong>
              </div>

              {selectedDocument && (
                <button
                  type="button"
                  className="scope-clear"
                  onClick={() => setSelectedDocument(null)}
                >
                  Search all
                </button>
              )}
            </div>

            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="chat-empty">
                  <div className="chat-empty-icon">
                    <MessageSquare size={24} />
                  </div>

                  <span className="chat-empty-eyebrow">
                    LET'S GET STARTED
                  </span>

                  <h4>What would you like to know?</h4>

                  <p>
                    Ask a question about your uploaded documents and get
                    an answer with source references.
                  </p>

                  <div className="suggestion-grid">
                    <button
                      type="button"
                      className="suggestion"
                      disabled={documents.length === 0}
                      onClick={() =>
                        useSuggestion(
                          "Summarize the key points in this document."
                        )
                      }
                    >
                      <span>
                        <FileText size={17} />
                      </span>

                      <span className="suggestion-copy">
                        <strong>Summarize key points</strong>
                        <small>Get a concise overview</small>
                      </span>

                      <ArrowUpRight
                        size={16}
                        className="suggestion-arrow"
                      />
                    </button>

                    <button
                      type="button"
                      className="suggestion"
                      disabled={documents.length === 0}
                      onClick={() =>
                        useSuggestion(
                          "What are the main topics covered?"
                        )
                      }
                    >
                      <span>
                        <FileText size={17} />
                      </span>

                      <span className="suggestion-copy">
                        <strong>Explore topics</strong>
                        <small>Discover the main ideas</small>
                      </span>

                      <ArrowUpRight
                        size={16}
                        className="suggestion-arrow"
                      />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="chat-conversation">
                  {messages.map((message, index) => (
                    <div
                      className={`message-row ${message.role}`}
                      key={`${message.role}-${index}`}
                    >
                      <div className={`message-avatar ${message.role}`}>
                        {message.role === "user" ? (
                          <UserRound size={17} />
                        ) : (
                          <Bot size={17} />
                        )}
                      </div>

                      <div className="message-content">
                        <span className="message-author">
                          {message.role === "user"
                            ? "You"
                            : message.role === "error"
                              ? "Something went wrong"
                              : "AI Assistant"}
                        </span>

                        <div className="message-bubble">
                          {message.role === "assistant"
                            ? renderAnswer(
                                message.text,
                                message.citations
                              )
                            : message.text}
                        </div>

                        {message.role === "assistant" &&
                          message.citations?.length > 0 && (
                            <div className="sources">
                              <button
                                type="button"
                                className="sources-toggle"
                                onClick={() => toggleSources(index)}
                                aria-expanded={!!expandedSources[index]}
                              >
                                {expandedSources[index] ? (
                                  <ChevronDown size={15} />
                                ) : (
                                  <ChevronRight size={15} />
                                )}

                                <FileText size={14} />

                                Sources ({message.citations.length})
                              </button>

                              {expandedSources[index] && (
                                <div className="source-list">
                                  {message.citations.map(
                                    (citation, citationIndex) => {
                                      const citationId = getCitationId(
                                        citation,
                                        citationIndex
                                      );

                                      const pageNumber =
                                        getPageNumber(citation);

                                      return (
                                        <button
                                          type="button"
                                          className="source-card source-card-button"
                                          key={`${citationId}-${citationIndex}`}
                                          onClick={() =>
                                            openCitation(
                                              citation,
                                              citationId
                                            )
                                          }
                                        >
                                          <div className="source-file-icon">
                                            <FileText size={17} />
                                          </div>

                                          <div className="source-info">
                                            <strong>
                                              {getFilename(citation)}
                                            </strong>

                                            <span>
                                              {citation.section_title
                                                ? `${citation.section_title} · `
                                                : ""}
                                              {pageNumber != null
                                                ? `Page ${pageNumber}`
                                                : "Page not specified"}
                                            </span>
                                          </div>

                                          <ExternalLink size={14} />
                                        </button>
                                      );
                                    }
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {asking && (
                <div className="message-row assistant">
                  <div className="message-avatar assistant">
                    <Bot size={17} />
                  </div>

                  <div className="message-content">
                    <span className="message-author">
                      AI Assistant
                    </span>

                    <div className="message-bubble thinking">
                      <LoaderCircle className="spin" size={17} />

                      <span>
                        Searching document sections and preparing an answer...
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <form className="chat-form" onSubmit={askQuestion}>
              <textarea
                id="question-input"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder={
                  documents.length
                    ? "Ask anything about your documents..."
                    : "Upload a PDF to start asking questions..."
                }
                rows={2}
                disabled={asking || documents.length === 0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
              />

              <div className="chat-form-bottom">
                <span>
                  <Sparkles size={12} />
                  Answers are grounded in retrieved document text.
                </span>

                <button
                  type="submit"
                  className="send-button"
                  disabled={
                    !question.trim() ||
                    asking ||
                    documents.length === 0
                  }
                  aria-label="Send question"
                  title="Send question"
                >
                  {asking ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <Send size={17} />
                  )}
                </button>
              </div>
            </form>

            <p className="chat-disclaimer">
              AI can make mistakes. Verify important information against
              the cited sources.
            </p>
          </section>
        </div>

        {/* PDF panel appears only after clicking a citation. */}
        {pdfViewer && renderPdfViewer()}
      </div>
    </div>
  );
}
