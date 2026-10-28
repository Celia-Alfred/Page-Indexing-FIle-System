
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

  const handleOpenChat = (documentId) => {
    const document = documents.find(
      (doc) =>
        (doc.document_id || doc.id || doc._id) === documentId
    );

    if (document) {
      setSelectedDocument(document);
    }

    setActivePage("chat");
  };

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        documents={documents}
        selectedDocument={
          selectedDocument
            ? selectedDocument.document_id ||
              selectedDocument.id ||
              selectedDocument._id
            : null
        }
        openChat={handleOpenChat}
      />

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="breadcrumb">Workspace / PageIndex</p>
            <h2>
              {activePage === "documents" ? "Documents" : "AI Chat"}
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
            selectedDocument={selectedDocument?.document_id || selectedDocument?.id || selectedDocument?._id || ""}
            selectedDoc={selectedDocument}
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
