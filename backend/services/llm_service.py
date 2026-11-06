
import json
import os
import time
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
).strip()

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from the backend .env file"
    )

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=90000,
    ),
)

MAX_ATTEMPTS = 4

RETRY_DELAYS = (2, 5, 10)

RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

NETWORK_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)

logger = logging.getLogger(__name__)



def _get_status_code(exc: Exception):
    """Safely extract an HTTP status code from a Gemini API error."""

    status_code = getattr(exc, "code", None)

    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _get_retry_delay(
    attempt: int,
    status_code: int | None = None,
) -> int:
    """Calculate retry delay based on the error type."""

    if status_code == 429:
        return 15 * (attempt + 1)

    if status_code is not None:
        return RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]

    return RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]



def generate_content_with_retry(
    contents: str,
    config: dict | None = None,
):
    """
    Generate content using Google Gemini.

    Retries:
    - HTTP 408
    - HTTP 429
    - HTTP 500, 502, 503, 504
    - Network connection failures
    - Connection timeouts
    - Remote protocol errors

    Returns a response object with:
    - response.text
    - response.raw_response
    """

    if not isinstance(contents, str) or not contents.strip():
        raise ValueError("Prompt cannot be empty")

    generation_config = {}

    if config:
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        supported_fields = (
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
            "response_mime_type",
            "response_schema",
        )

        for field in supported_fields:
            if field in config:
                generation_config[field] = config[field]

    gemini_config = (
        types.GenerateContentConfig(**generation_config)
        if generation_config
        else None
    )

    for attempt in range(MAX_ATTEMPTS):

        try:
            logger.info(
                "Sending Gemini request using model %s "
                "(attempt %s/%s)",
                GEMINI_MODEL,
                attempt + 1,
                MAX_ATTEMPTS,
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=gemini_config,
            )

            generated_text = (
                getattr(response, "text", None) or ""
            ).strip()

            if not generated_text:
                raise RuntimeError(
                    "Gemini returned an empty response"
                )

            logger.info("Gemini response received successfully")

            return SimpleNamespace(
                text=generated_text,
                raw_response=response,
            )



        except errors.APIError as exc:

            status_code = _get_status_code(exc)

            if status_code not in RETRYABLE_STATUS_CODES:
                logger.exception(
                    "Non-retryable Gemini API error: %s",
                    exc,
                )
                raise

            if attempt == MAX_ATTEMPTS - 1:
                logger.exception(
                    "Gemini API failed after all retry attempts"
                )
                raise

            delay = _get_retry_delay(
                attempt,
                status_code,
            )

            logger.warning(
                "Gemini API error %s: %s. "
                "Retrying in %s seconds (%s/%s)",
                status_code,
                exc,
                delay,
                attempt + 2,
                MAX_ATTEMPTS,
            )

            time.sleep(delay)



        except NETWORK_EXCEPTIONS as exc:

            if attempt == MAX_ATTEMPTS - 1:
                logger.exception(
                    "Gemini network request failed "
                    "after all retry attempts"
                )
                raise

            delay = _get_retry_delay(attempt)

            logger.warning(
                "Gemini network error: %s. "
                "Retrying in %s seconds (%s/%s)",
                exc,
                delay,
                attempt + 2,
                MAX_ATTEMPTS,
            )

            time.sleep(delay)


        except Exception:
            logger.exception(
                "Unexpected error while generating "
                "Gemini content"
            )
            raise

    raise RuntimeError(
        "Gemini request failed after all retry attempts"
    )




def generate_summary(text: str) -> str:
    """
    Generate a concise, document-grounded summary.
    Used for hierarchical document indexing.
    """

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
        raise RuntimeError(
            "Gemini returned an empty summary"
        )

    return summary



def understand_query(question: str) -> dict:
    """
    Analyze a question for hierarchical,
    vectorless document retrieval.
    """

    if not isinstance(question, str) or not question.strip():
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
        config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
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
        or not all(
            isinstance(keyword, str)
            for keyword in keywords
        )
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
