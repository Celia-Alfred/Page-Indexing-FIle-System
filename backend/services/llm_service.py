import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from the backend .env file"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

MAX_ATTEMPTS = 3
RETRY_DELAYS = (2, 4)


def generate_content_with_retry(
    contents: str,
    config: dict | None = None,
):
    """Retry Gemini requests up to 3 times for temporary 503 errors."""

    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )

        except errors.ServerError as exc:
            if exc.code != 503 or attempt == MAX_ATTEMPTS - 1:
                raise

            delay = RETRY_DELAYS[attempt]
            print(
                f"Gemini returned 503. "
                f"Retrying in {delay} seconds "
                f"(attempt {attempt + 2}/{MAX_ATTEMPTS})..."
            )
            time.sleep(delay)


def generate_summary(text: str) -> str:
    """Generate a concise, document-grounded summary using Gemini."""

    if not text or not text.strip():
        return "No text available to summarize."

    prompt = f"""
You are summarizing a section of a document for a hierarchical
document index.

Create a concise summary that:
- Preserves the main topics, facts, names, and technical terms.
- Includes important details useful for answering questions later.
- Does not introduce facts that are absent from the source.
- Uses clear, factual language.

Return only the summary.

DOCUMENT SECTION:
{text}
"""

    response = generate_content_with_retry(prompt)

    summary = (response.text or "").strip()

    if not summary:
        raise RuntimeError("Gemini returned an empty summary")

    return summary


def understand_query(question: str) -> dict:
    """Analyze a question for hierarchical, vectorless retrieval."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    prompt = f"""
You are the query-understanding component of a vectorless,
hierarchical document retrieval system.

Analyze the user's question and return a JSON object containing
exactly these fields:

1. topic: A string describing the main subject of the question.
2. keywords: A list of important search terms from the question.
3. information_needed: A string describing what information
   is required to answer the question.

Rules:
- Do not answer the question.
- Do not assume facts about any document.
- Extract useful search terms from the question.
- Preserve names, technologies, and important phrases.
- Return valid JSON only, without Markdown fences.

Example output:
{{
    "topic": "backend technologies",
    "keywords": ["backend", "technologies", "worked with"],
    "information_needed": "Identify the backend technologies mentioned in the relevant document sections."
}}

USER QUESTION:
{question}
"""

    response = generate_content_with_retry(
        prompt,
        config={"response_mime_type": "application/json"},
    )

    result = (response.text or "").strip()

    if not result:
        raise RuntimeError(
            "Gemini returned an empty query analysis"
        )

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON for query understanding"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Gemini returned an unexpected query format"
        )

    topic = parsed.get("topic")
    keywords = parsed.get("keywords")
    information_needed = parsed.get("information_needed")

    if (
        not isinstance(topic, str)
        or not topic.strip()
        or not isinstance(keywords, list)
        or not all(isinstance(keyword, str) for keyword in keywords)
        or not isinstance(information_needed, str)
        or not information_needed.strip()
    ):
        raise RuntimeError(
            "Gemini returned an unexpected query format"
        )

    return {
        "topic": topic.strip(),
        "keywords": keywords,
        "information_needed": information_needed.strip(),
    }