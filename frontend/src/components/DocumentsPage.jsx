import { useMemo, useRef, useState } from "react";
import {
  Upload,
  LoaderCircle,
  FileText,
  Search,
  X,
  ArrowUpDown,
} from "lucide-react";

const API_BASE_URL = "http://127.0.0.1:8000";
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function DocumentsPage({
  documents = [],
  setDocuments = () => {},
  setActivePage = () => {},
  setSelectedDocument = () => {},
}) {
  const fileInputRef = useRef(null);

  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");

  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("newest");

  const safeDocuments = Array.isArray(documents) ? documents : [];

  const filteredDocuments = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    const filtered = safeDocuments.filter((document) => {
      const name =
        document.filename ||
        document.file_name ||
        document.name ||
        "Untitled document";

      return name.toLowerCase().includes(query);
    });

    return [...filtered].sort((a, b) => {
      const nameA = (
        a.filename ||
        a.file_name ||
        a.name ||
        ""
      ).toLowerCase();

      const nameB = (
        b.filename ||
        b.file_name ||
        b.name ||
        ""
      ).toLowerCase();

      if (sortBy === "name-asc") {
        return nameA.localeCompare(nameB);
      }

      if (sortBy === "name-desc") {
        return nameB.localeCompare(nameA);
      }

      const dateA = new Date(
        a.uploadedAt || a.uploaded_at || a.created_at || 0
      ).getTime();

      const dateB = new Date(
        b.uploadedAt || b.uploaded_at || b.created_at || 0
      ).getTime();

      return sortBy === "oldest"
        ? dateA - dateB
        : dateB - dateA;
    });
  }, [safeDocuments, searchQuery, sortBy]);

  const uploadPdf = async (file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Please select a PDF file.");
      setUploadMessage("");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setUploadError("PDF size must not exceed 10 MB.");
      setUploadMessage("");
      return;
    }

    if (uploading) return;

    setUploading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          result.detail || `Upload failed (${response.status}).`
        );
      }

      const uploadedDocument = {
        ...result,
        id: result.id || result.document_id || result._id,
        filename:
          result.filename || result.file_name || result.name || file.name,
        name: result.name || result.filename || file.name,
        size: file.size,
        uploadedAt: new Date().toISOString(),
      };

      setDocuments((previousDocuments) => {
        const existingDocuments = Array.isArray(previousDocuments)
          ? previousDocuments
          : [];

        return [
          uploadedDocument,
          ...existingDocuments.filter(
            (document) =>
              document.id !== uploadedDocument.id ||
              !uploadedDocument.id
          ),
        ];
      });

      setUploadMessage(
        result.message || `${file.name} uploaded successfully.`
      );

      setSearchQuery("");
      setSortBy("newest");
    } catch (error) {
      console.error("PDF upload error:", error);

      setUploadError(
        error.message ||
          "Could not upload the PDF. Check that the backend is running."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleSelectedFile = (file) => {
    if (!file) return;

    setUploadError("");
    setUploadMessage("");
    uploadPdf(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    if (uploading) return;

    const files = Array.from(event.dataTransfer.files || []);

    if (files.length === 0) return;

    if (files.length > 1) {
      setUploadError("Please upload one PDF at a time.");
      setUploadMessage("");
      return;
    }

    handleSelectedFile(files[0]);
  };

  const openDocument = (document) => {
    setSelectedDocument(document);
    setActivePage("chat");
  };

  return (
    <section className="documents-page">
      <div className="page-heading">
        <div>
          <h2>Your Documents</h2>
          <p>Upload PDFs to build their intelligent page index.</p>
        </div>

        <span className="document-count">
          {safeDocuments.length}{" "}
          {safeDocuments.length === 1 ? "document" : "documents"}
        </span>
      </div>

      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${
          uploading ? "uploading" : ""
        }`}
        onDragEnter={(event) => {
          event.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();

          if (
            !event.relatedTarget ||
            !event.currentTarget.contains(event.relatedTarget)
          ) {
            setDragging(false);
          }
        }}
        onDrop={handleDrop}
      >
        <div className="upload-icon">
          {uploading ? (
            <LoaderCircle className="spin" size={27} />
          ) : (
            <Upload size={27} />
          )}
        </div>

        <h4>
          {uploading
            ? "Indexing your document..."
            : dragging
              ? "Drop your PDF here"
              : "Drag and drop your PDF here"}
        </h4>

        <p>
          {uploading
            ? "Please wait while your document is processed."
            : "Or browse files from your computer"}
        </p>

        <button
          className="primary-button"
          type="button"
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={16} />
          Choose PDF
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          hidden
          disabled={uploading}
          onChange={(event) => {
            handleSelectedFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />

        <span className="file-hint">
          PDF files only · Maximum 10 MB
        </span>
      </div>

      {uploading && (
        <div className="upload-status" role="status">
          <LoaderCircle className="spin" size={18} />
          Uploading and indexing PDF...
        </div>
      )}

      {uploadError && (
        <div className="upload-error" role="alert">
          {uploadError}
        </div>
      )}

      {uploadMessage && (
        <div className="upload-success" role="status">
          {uploadMessage}
        </div>
      )}

      <div className="documents-list">
        <div className="section-heading documents-toolbar-heading">
          <div>
            <h3>Indexed Documents</h3>

            <p className="results-count">
              Showing {filteredDocuments.length} of {safeDocuments.length}{" "}
              {safeDocuments.length === 1 ? "document" : "documents"}
            </p>
          </div>
        </div>

        <div className="document-filters">
          <div className="document-search">
            <Search size={18} />

            <input
              type="text"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search documents by filename..."
              aria-label="Search documents by filename"
            />

            {searchQuery && (
              <button
                type="button"
                className="clear-search"
                onClick={() => setSearchQuery("")}
                aria-label="Clear search"
                title="Clear search"
              >
                <X size={17} />
              </button>
            )}
          </div>

          <label className="document-sort">
            <ArrowUpDown size={17} />

            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value)}
              aria-label="Sort documents"
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="name-asc">Name: A to Z</option>
              <option value="name-desc">Name: Z to A</option>
            </select>
          </label>
        </div>

        {safeDocuments.length === 0 ? (
          <div className="empty-state">
            <FileText size={34} />
            <h4>No documents yet</h4>
            <p>Upload a PDF to see it listed here.</p>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="empty-state no-search-results">
            <Search size={34} />
            <h4>No matching documents</h4>

            <p>
              No filenames match "{searchQuery}". Try another search term.
            </p>

            <button
              className="secondary-button"
              type="button"
              onClick={() => setSearchQuery("")}
            >
              Clear search
            </button>
          </div>
        ) : (
          <div className="document-grid">
            {filteredDocuments.map((document, index) => {
              const documentName =
                document.filename ||
                document.file_name ||
                document.name ||
                "Untitled document";

              const documentKey =
                document.id ||
                document.document_id ||
                document._id ||
                `${documentName}-${index}`;

              return (
                <article className="document-card" key={documentKey}>
                  <div className="document-icon">
                    <FileText size={24} />
                  </div>

                  <div className="document-details">
                    <h4 title={documentName}>{documentName}</h4>

                    <p>
                      {typeof document.size === "number"
                        ? `${(document.size / (1024 * 1024)).toFixed(2)} MB`
                        : "PDF document"}
                    </p>
                  </div>

                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => openDocument(document)}
                  >
                    Ask AI
                  </button>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

export default DocumentsPage;