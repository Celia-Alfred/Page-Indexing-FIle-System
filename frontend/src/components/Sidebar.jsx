
import {
  Database,
  FileText,
  MessageSquare,
} from "lucide-react";

export default function Sidebar({
  activePage = "documents",
  setActivePage = () => {},
  documents = [],
  selectedDocument = null,
  openChat = () => {},
}) {
  const safeDocuments = Array.isArray(documents) ? documents : [];

  const getDocumentId = (doc) =>
    doc.document_id || doc.id || doc._id;

  const getDocumentName = (doc) =>
    doc.filename || doc.file_name || doc.name || "Untitled document";

  const handleOpenChat = (doc) => {
    const documentId = getDocumentId(doc);

    if (typeof openChat === "function") {
      openChat(documentId);
    }

    if (typeof setActivePage === "function") {
      setActivePage("chat");
    }
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Database size={22} />
        </div>

        <div>
          <h1>PageIndex</h1>
          <span>Document intelligence</span>
        </div>
      </div>

      <div className="sidebar-label">WORKSPACE</div>

      <button
        className={`nav-item ${
          activePage === "documents" ? "active" : ""
        }`}
        type="button"
        onClick={() => setActivePage("documents")}
      >
        <FileText size={18} />
        <span>My documents</span>
        <span className="nav-count">{safeDocuments.length}</span>
      </button>

      <button
        className={`nav-item ${
          activePage === "chat" ? "active" : ""
        }`}
        type="button"
        onClick={() => setActivePage("chat")}
      >
        <MessageSquare size={18} />
        <span>Ask documents</span>
      </button>

      <div className="sidebar-divider" />

      <div className="sidebar-label">YOUR LIBRARY</div>

      <div className="sidebar-documents">
        {safeDocuments.length === 0 ? (
          <p className="empty-library">
            Your uploaded PDFs will appear here.
          </p>
        ) : (
          safeDocuments.map((doc, index) => {
            const documentId = getDocumentId(doc);
            const documentName = getDocumentName(doc);

            return (
              <button
                type="button"
                key={documentId || `${documentName}-${index}`}
                className={`library-item ${
                  selectedDocument === documentId ? "selected" : ""
                }`}
                onClick={() => handleOpenChat(doc)}
                title={documentName}
              >
                <FileText size={17} />
                <span>{documentName}</span>
              </button>
            );
          })
        )}
      </div>

      <div className="sidebar-footer">
        <span className="status-dot" />
        <span>Vectorless retrieval</span>
      </div>
    </aside>
  );
}
