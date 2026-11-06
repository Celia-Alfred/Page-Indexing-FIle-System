
import { useState } from "react";

import Sidebar from "./components/Sidebar";
import DocumentsPage from "./components/DocumentsPage";
import ChatPage from "./components/ChatPage";

import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("documents");
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [messages, setMessages] = useState([]);

  const getDocumentId = (doc) => {
    if (!doc) return null;

    if (typeof doc === "string") {
      return doc;
    }

    return (
      doc.document_id ||
      doc.id ||
      doc._id ||
      null
    );
  };

  const selectedDocumentId = getDocumentId(selectedDocument);

  const selectedDoc =
    documents.find(
      (doc) =>
        String(getDocumentId(doc)) ===
        String(selectedDocumentId)
    ) || null;

  const handleOpenChat = (documentId) => {
    const normalizedId = getDocumentId(documentId);

    if (normalizedId) {
      setSelectedDocument(String(normalizedId));
    } else {
      setSelectedDocument(null);
    }

    setActivePage("chat");
  };

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        documents={documents}
        selectedDocument={selectedDocumentId}
        openChat={handleOpenChat}
      />

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="breadcrumb">
              Workspace / PageIndex
            </p>

            <h2>
              {activePage === "documents"
                ? "Documents"
                : "AI Chat"}
            </h2>
          </div>

          <div className="system-status">
            <span className="status-dot" />
            PageIndex Workspace
          </div>
        </header>

        {activePage === "documents" ? (
          <DocumentsPage
            documents={documents}
            setDocuments={setDocuments}
            setActivePage={setActivePage}
            setSelectedDocument={setSelectedDocument}
          />
        ) : (
          <ChatPage
            documents={documents}
            selectedDocument={selectedDocumentId}
            selectedDoc={selectedDoc}
            setSelectedDocument={setSelectedDocument}
            messages={messages}
            setMessages={setMessages}
            setActivePage={setActivePage}
          />
        )}

        <footer className="app-footer">
          PageIndex · Intelligent Document Indexing
        </footer>
      </main>
    </div>
  );
}

export default App;
