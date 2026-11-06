
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
  BookOpen,
  X,
} from "lucide-react";

import { queryDocument } from "../services/api";
import PdfViewer from "./PdfViewer";

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
  const [activeCitation, setActiveCitation] = useState(null);

  const messagesEndRef = useRef(null);

  const safeDocuments = Array.isArray(documents)
    ? documents
    : [];

  const safeMessages = Array.isArray(messages)
    ? messages
    : [];

  function normalizeDocumentId(document) {
    if (!document) return "";

    if (typeof document === "string") {
      return document.trim();
    }

    const id =
      document.document_id ||
      document.id ||
      document._id?.$oid ||
      document._id ||
      "";

    return id ? String(id).trim() : "";
  }

  const selectedDocumentId =
    normalizeDocumentId(selectedDocument);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, asking]);
  async function askQuestion(event) {
    event.preventDefault();

    const questionText = question.trim();

    if (!questionText || asking) return;
    if (safeDocuments.length === 0) return;
    if (typeof setMessages !== "function") return;

    setMessages((previous) => [
      ...(Array.isArray(previous) ? previous : []),
      {
        role: "user",
        text: questionText,
      },
    ]);

    setQuestion("");
    setActiveCitation(null);
    setAsking(true);

    try {
      const data = await queryDocument({
        question: questionText,
        documentId: selectedDocumentId || null,
        topK: 3,
      });

      setMessages((previous) => [
        ...(Array.isArray(previous) ? previous : []),
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
          queryAnalysis: data.query_analysis,
        },
      ]);
    } catch (error) {
      console.error("Document query error:", error);

      const errorMessage =
        error?.message || "An unexpected error occurred.";

      setMessages((previous) => [
        ...(Array.isArray(previous) ? previous : []),
        {
          role: "error",
          text: errorMessage.includes("Failed to fetch")
            ? "Cannot reach the backend. Check that FastAPI is running and CORS is configured."
            : errorMessage,
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

  function getCitationId(citation, index) {
    const rawId =
      citation?.source_id ??
      citation?.citation_id ??
      citation?.id;

    if (rawId == null || rawId === "") {
      return `S${index + 1}`;
    }

    const normalizedId = String(rawId)
      .trim()
      .replace(/^\[/, "")
      .replace(/\]$/, "");

    if (/^\d+$/.test(normalizedId)) {
      return `S${normalizedId}`;
    }

    return normalizedId;
  }

  function getPageNumber(citation) {
    return (
      citation?.page_number ??
      citation?.page ??
      null
    );
  }

  function getDocumentId(citation) {
    const rawId =
      citation?.document_id ||
      citation?.documentId ||
      "";

    return rawId ? String(rawId).trim() : "";
  }

  function getDocumentForCitation(citation) {
    const documentId = getDocumentId(citation);

    const filename = (
      citation?.filename ||
      citation?.file_name ||
      ""
    ).trim().toLowerCase();

    if (documentId) {
      const matchingById = safeDocuments.find(
        (doc) => normalizeDocumentId(doc) === documentId
      );

      if (matchingById) {
        return matchingById;
      }
    }

    if (filename) {
      return safeDocuments.find((doc) => {
        const docFilename = (
          doc.filename ||
          doc.file_name ||
          doc.name ||
          ""
        ).trim().toLowerCase();

        return docFilename === filename;
      });
    }

    return null;
  }

  function openCitation(citation, citationId) {
    const matchingDocument =
      getDocumentForCitation(citation);

    const resolvedDocumentId =
      normalizeDocumentId(matchingDocument) ||
      getDocumentId(citation);

    if (!resolvedDocumentId) {
      console.error(
        "Citation document ID could not be resolved:",
        {
          citation,
          matchingDocument,
          availableDocuments: safeDocuments,
        }
      );

      return;
    }

    const resolvedCitation = {
      ...citation,
      citationId,
      document_id: resolvedDocumentId,
      filename:
        citation.filename ||
        citation.file_name ||
        matchingDocument?.filename ||
        matchingDocument?.file_name ||
        matchingDocument?.name ||
        "Source document",
    };

    console.log("Opening citation PDF:", {
      documentId: resolvedDocumentId,
      filename: resolvedCitation.filename,
      page: getPageNumber(resolvedCitation),
    });

    setActiveCitation(resolvedCitation);

    if (typeof setSelectedDocument === "function") {
      setSelectedDocument(resolvedDocumentId);
    }
  }

  function closeCitation() {
    setActiveCitation(null);
  }

  function renderAnswer(text, citations = []) {
    if (!text) return null;

    const safeCitations = Array.isArray(citations)
      ? citations
      : [];

    const citationMap = new Map(
      safeCitations.map((citation, index) => [
        getCitationId(citation, index),
        citation,
      ])
    );

    const parts = String(text).split(/(\[S\d+\])/g);

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

      const filename =
        citation.filename ||
        citation.file_name ||
        "Source document";

      return (
        <button
          key={`citation-${index}`}
          type="button"
          className="inline-citation"
          title={`Open ${filename}, page ${pageNumber ?? "unknown"}`}
          onClick={() =>
            openCitation(citation, match[1])
          }
        >
          {filename} · Page {pageNumber ?? "—"}
        </button>
      );
    });
  }

  function toggleSources(messageIndex) {
    setExpandedSources((previous) => ({
      ...previous,
      [messageIndex]: !previous[messageIndex],
    }));
  }

  function renderCitationDetails(citation) {
    if (!citation) return null;

    const pageNumber = getPageNumber(citation);

    const evidence = Array.isArray(citation.evidence)
      ? citation.evidence
      : [];

    const excerpt =
      citation.excerpt ||
      evidence
        .map((item) =>
          item.excerpt || item.quote || item.text || ""
        )
        .filter(Boolean)
        .join("\n\n");

    return (
      <div className="chat-pdf-citation-details">
        <h4>
          <BookOpen
            size={14}
            style={{
              display: "inline",
              verticalAlign: "middle",
              marginRight: 6,
            }}
          />
          Cited passage
        </h4>

        <p>
          {excerpt ||
            "No extracted passage was provided for this citation."}
        </p>

        <div className="citation-detail-meta">
          {pageNumber != null && (
            <span>Page {pageNumber}</span>
          )}

          {citation.section_title && (
            <span>{citation.section_title}</span>
          )}

          {citation.extraction_source && (
            <span>
              {citation.extraction_source.toUpperCase()} extraction
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="chat-page">
      {/* Welcome section */}
      <section className="welcome-section chat-welcome">
        <div className="welcome-copy">
          <div className="eyebrow">
            <Sparkles size={14} />
            YOUR DOCUMENTS, YOUR ANSWERS
          </div>

          <h2>
            What do you want to{" "}
            <span>discover?</span>
          </h2>

          <p>
            Ask questions and get AI-generated answers
            grounded in your indexed PDF content.
          </p>
        </div>
      </section>

      {/* Empty document notice */}
      {safeDocuments.length === 0 && (
        <div className="chat-no-documents">
          <FileText size={19} />

          <div className="chat-no-documents-copy">
            <strong>No documents uploaded yet</strong>

            <span>
              Upload a PDF before asking questions
              about your documents.
            </span>
          </div>

          <button
            type="button"
            className="chat-upload-link"
            onClick={() => setActivePage("documents")}
          >
            Upload PDF
            <ArrowUpRight size={15} />
          </button>
        </div>
      )}

      {/* Chat workspace */}
      <div
        className={`chat-workspace ${
          activeCitation ? "pdf-open" : ""
        }`}
      >
        {/* Chat panel */}
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

          {/* Document search scope */}
          <div className="chat-scope">
            <div className="chat-scope-left">
              <Search size={15} />

              <span className="scope-label">
                Searching in
              </span>

              <strong>
                {selectedDocumentId
                  ? selectedDoc?.filename ||
                    selectedDoc?.file_name ||
                    selectedDoc?.name ||
                    safeDocuments.find(
                      (doc) =>
                        normalizeDocumentId(doc) ===
                        selectedDocumentId
                    )?.filename ||
                    "Selected document"
                  : "All indexed documents"}
              </strong>
            </div>

            {selectedDocumentId && (
              <button
                type="button"
                className="scope-clear"
                onClick={() => {
                  setSelectedDocument(null);
                  setActiveCitation(null);
                }}
              >
                Search all
              </button>
            )}
          </div>

          {/* Chat messages */}
          <div className="chat-messages">
            {safeMessages.length === 0 ? (
              <div className="chat-empty">
                <div className="chat-empty-icon">
                  <MessageSquare size={24} />
                </div>

                <span className="chat-empty-eyebrow">
                  LET'S GET STARTED
                </span>

                <h4>What would you like to know?</h4>

                <p>
                  Ask a question about your uploaded
                  documents and get an answer with
                  source references.
                </p>

                <div className="suggestion-grid">
                  <button
                    type="button"
                    className="suggestion"
                    disabled={safeDocuments.length === 0}
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
                    disabled={safeDocuments.length === 0}
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
                {safeMessages.map((message, index) => (
                  <div
                    className={`message-row ${message.role}`}
                    key={`${message.role}-${index}`}
                  >
                    <div
                      className={`message-avatar ${message.role}`}
                    >
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

                      {/* Citation sources */}
                      {message.role === "assistant" &&
                        message.citations?.length > 0 && (
                          <div className="sources">
                            <button
                              type="button"
                              className="sources-toggle"
                              onClick={() =>
                                toggleSources(index)
                              }
                              aria-expanded={
                                !!expandedSources[index]
                              }
                            >
                              {expandedSources[index] ? (
                                <ChevronDown size={15} />
                              ) : (
                                <ChevronRight size={15} />
                              )}

                              <FileText size={14} />

                              Sources (
                              {message.citations.length})
                            </button>

                            {expandedSources[index] && (
                              <div className="source-list">
                                {message.citations.map(
                                  (citation, citationIndex) => {
                                    const citationId =
                                      getCitationId(
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
                                            {citation.filename ||
                                              citation.file_name ||
                                              "Source document"}
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

                                        <BookOpen size={15} />
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

            {/* Loading indicator */}
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
                    <LoaderCircle
                      className="spin"
                      size={17}
                    />

                    <span>
                      Searching document sections and
                      preparing an answer...
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Question input */}
          <form
            className="chat-form"
            onSubmit={askQuestion}
          >
            <textarea
              id="question-input"
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder={
                safeDocuments.length
                  ? "Ask anything about your documents..."
                  : "Upload a PDF to start asking questions..."
              }
              rows={2}
              disabled={
                asking || safeDocuments.length === 0
              }
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey
                ) {
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
                  safeDocuments.length === 0
                }
                aria-label="Send question"
                title="Send question"
              >
                {asking ? (
                  <LoaderCircle
                    className="spin"
                    size={17}
                  />
                ) : (
                  <Send size={17} />
                )}
              </button>
            </div>
          </form>

          <p className="chat-disclaimer">
            AI can make mistakes. Verify important
            information against the cited sources.
          </p>
        </section>

        {/* PDF viewer - appears only after citation click */}
        {activeCitation && (
          <aside className="chat-pdf-panel">
            <div className="chat-pdf-header">
              <div>
                <h3>
                  {activeCitation.filename ||
                    "Source document"}
                </h3>

                <p>
                  Page {getPageNumber(activeCitation) || "—"}
                  {" · "}
                  {activeCitation.extraction_source
                    ? `${activeCitation.extraction_source.toUpperCase()} extraction`
                    : "Verified source"}
                </p>
              </div>

              <button
                type="button"
                className="citation-close"
                onClick={closeCitation}
                aria-label="Close PDF viewer"
                title="Close PDF viewer"
              >
                <X size={18} />
              </button>
            </div>

            <PdfViewer
              documentId={activeCitation.document_id}
              initialPage={getPageNumber(activeCitation) || 1}
              citation={activeCitation}
            />

            {renderCitationDetails(activeCitation)}
          </aside>
        )}
      </div>
    </div>
  );
}
