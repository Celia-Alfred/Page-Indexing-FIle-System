
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
  BookOpen,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function askQuestion(event) {
    event.preventDefault();

    const questionText = question.trim();
    if (!questionText || asking) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", text: questionText },
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
        headers: { "Content-Type": "application/json" },
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
          sources: Array.isArray(data.sources) ? data.sources : [],
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

  function getCitationId(citation, index) {
    return String(
      citation.source_id ||
      citation.id ||
      `S${index + 1}`
    );
  }

  function getPageNumber(citation) {
    return citation.page_number ?? citation.page ?? null;
  }

  function getDocumentId(citation) {
    return (
      citation.document_id ||
      citation.documentId ||
      ""
    );
  }

  function getDocumentForCitation(citation) {
    const documentId = getDocumentId(citation);
    const filename = citation.filename || citation.file_name;

    return documents.find((doc) => {
      const id = doc.document_id || doc.id || doc._id;

      return (
        (documentId && String(id) === String(documentId)) ||
        (!documentId && filename && doc.filename === filename)
      );
    });
  }

  function openCitation(citation, citationId) {
    setActiveCitation({
      ...citation,
      citationId,
    });

    const document = getDocumentForCitation(citation);

    if (document) {
      setSelectedDocument(document);
    }

  }

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

      return (
        <button
          key={`citation-${index}`}
          type="button"
          className="inline-citation"
          title={
            pageNumber
              ? `View source details for page ${pageNumber}`
              : "View source details"
          }
          onClick={() => openCitation(citation, match[1])}
        >
          {pageNumber ? `[Page ${pageNumber}]` : `[${match[1]}]`}
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

  function renderSourceDetails(citation) {
    if (!citation) return null;

    const pageNumber = getPageNumber(citation);
    const filename = citation.filename || citation.file_name;
    const documentId = getDocumentId(citation);
    const document = getDocumentForCitation(citation);

    return (
      <div className="citation-detail-panel">
        <div className="citation-detail-heading">
          <div className="citation-detail-icon">
            <BookOpen size={17} />
          </div>

          <div>
            <strong>Source details</strong>
            <span>{filename || "Referenced document"}</span>
          </div>

          <button
            type="button"
            className="citation-close"
            onClick={() => setActiveCitation(null)}
            aria-label="Close source details"
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="citation-detail-meta">
          {pageNumber != null && (
            <span>Page {pageNumber}</span>
          )}

          {citation.section_title && (
            <span>{citation.section_title}</span>
          )}

          {citation.section && (
            <span>{citation.section}</span>
          )}
        </div>

        {citation.excerpt && (
          <div className="citation-excerpt">
            <span className="citation-excerpt-label">Cited passage</span>
            <p>{citation.excerpt}</p>
          </div>
        )}

        {document && (
          <button
            type="button"
            className="citation-open-document"
            onClick={() => {
              setSelectedDocument(document);
              setActivePage("documents");
            }}
          >
            <FileText size={15} />
            View document in workspace
            <ExternalLink size={14} />
          </button>
        )}

        {!document && documentId && (
          <p className="citation-note">
            The document is referenced by ID, but it isn't currently
            available in the loaded document list.
          </p>
        )}

        {!pageNumber && (
          <p className="citation-note">
            The backend did not return a specific page number for this source.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="chat-page">
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
            Ask questions and get AI-generated answers grounded
            in your indexed PDF content.
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
                    useSuggestion("Summarize the key points in this document.")
                  }
                >
                  <span><FileText size={17} /></span>
                  <span className="suggestion-copy">
                    <strong>Summarize key points</strong>
                    <small>Get a concise overview</small>
                  </span>
                  <ArrowUpRight size={16} className="suggestion-arrow" />
                </button>

                <button
                  type="button"
                  className="suggestion"
                  disabled={documents.length === 0}
                  onClick={() =>
                    useSuggestion("What are the main topics covered?")
                  }
                >
                  <span><FileText size={17} /></span>
                  <span className="suggestion-copy">
                    <strong>Explore topics</strong>
                    <small>Discover the main ideas</small>
                  </span>
                  <ArrowUpRight size={16} className="suggestion-arrow" />
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
                        ? renderAnswer(message.text, message.citations)
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
                              {message.citations.map((citation, citationIndex) => {
                                const citationId = getCitationId(
                                  citation,
                                  citationIndex
                                );
                                const pageNumber = getPageNumber(citation);

                                return (
                                  <button
                                    type="button"
                                    className="source-card source-card-button"
                                    key={`${citationId}-${citationIndex}`}
                                    onClick={() =>
                                      openCitation(citation, citationId)
                                    }
                                  >
                                    <div className="source-file-icon">
                                      <FileText size={17} />
                                    </div>

                                    <div className="source-info">
                                      <strong>
                                        {citation.filename || "Source document"}
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
                              })}
                            </div>
                          )}
                        </div>
                      )}

                    {message.role === "assistant" &&
                      activeCitation && (
                        <div className="citation-detail-wrapper">
                          {renderSourceDetails(activeCitation)}
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
                <span className="message-author">AI Assistant</span>
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
              disabled={!question.trim() || asking || documents.length === 0}
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
          AI can make mistakes. Verify important information against the cited sources.
        </p>
      </section>
    </div>
  );
}
