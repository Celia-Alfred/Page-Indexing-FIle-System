
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function handleResponse(response) {
  const result = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      result.detail || `Request failed (${response.status}).`
    );
  }

  return result;
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/documents/upload`,
    {
      method: "POST",
      body: formData,
    }
  );

  return handleResponse(response);
}

export async function queryDocument({
  question,
  documentId,
  topK = 3,
}) {
  const response = await fetch(
    `${API_BASE_URL}/documents/query`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        document_id: documentId || null,
        top_k: topK,
      }),
    }
  );

  return handleResponse(response);
}

export function getDocumentPdfUrl(documentId) {
  if (!documentId) {
    throw new Error("Document ID is required.");
  }

  return `${API_BASE_URL}/documents/${encodeURIComponent(
    documentId
  )}/pdf`;
}
