"""
V6.1 External Regulatory Research

Controlled, read-only search helper for Indonesian regulatory sources.
This module does not modify the internal RAG/vectorstore.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import logging
import re


# Official / trusted regulatory domains. Keep this allowlist conservative.
logger = logging.getLogger(__name__)


OFFICIAL_DOMAINS = {
    "peraturan.bpk.go.id": {
        "source_name": "JDIH BPK",
        "source_tier": 1,
    },
    "jdihn.go.id": {
        "source_name": "JDIHN",
        "source_tier": 1,
    },
}


def _normalize_domain(url: str) -> str:
    """Return a normalized hostname without www."""
    hostname = urlparse(url).hostname or ""
    return hostname.lower().removeprefix("www.")


def _build_bpk_search_url(query: str) -> str:
    """Build a controlled JDIH BPK search URL."""
    return f"https://peraturan.bpk.go.id/Search?query={quote(query)}"


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(1))).strip()


def _strip_html(html: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch(url: str, timeout: int = 10) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Regula/6.3 (+controlled-regulatory-research)"
            )
        },
    )

    logger.info("External research fetch: url=%s", url)

    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        logger.info(
            "External research response: status=%s content_type=%s",
            getattr(response, "status", "unknown"),
            content_type,
        )
        if "text/html" not in content_type.lower():
            logger.warning(
                "External research skipped non-HTML response: content_type=%s",
                content_type,
            )
            return ""
        body = response.read().decode("utf-8", errors="replace")
        logger.info("External research body received: chars=%d", len(body))
        return body


def fetch_external_source(
    result: Dict[str, Any],
    timeout: int = 10,
    max_chars: int = 20000,
) -> Dict[str, Any]:
    """
    Fetch one validated external source and return bounded text evidence.

    The source must pass V6.2 validation first. No files are downloaded,
    persisted, or inserted into the internal RAG/vectorstore.
    """
    if not validate_external_source(result):
        return {
            "status": "invalid_source",
            "source": result,
            "evidence": "",
        }

    max_chars = max(1000, min(int(max_chars), 50000))
    url = str(result["url"]).strip()

    try:
        html = _fetch(url, timeout=timeout)
    except HTTPError as exc:
        logger.warning(
            "External source HTTP error: code=%s reason=%s url=%s",
            exc.code,
            exc.reason,
            url,
        )
        return {
            "status": "fetch_failed",
            "source": result,
            "evidence": "",
        }
    except URLError as exc:
        logger.warning(
            "External source URL error: reason=%s url=%s",
            exc.reason,
            url,
        )
        return {
            "status": "fetch_failed",
            "source": result,
            "evidence": "",
        }
    except (TimeoutError, ValueError) as exc:
        logger.warning(
            "External source fetch error: type=%s detail=%s url=%s",
            type(exc).__name__,
            exc,
            url,
        )
        return {
            "status": "fetch_failed",
            "source": result,
            "evidence": "",
        }

    text = _strip_html(html)
    if not text:
        return {
            "status": "empty_source",
            "source": result,
            "evidence": "",
        }

    evidence = text[:max_chars]

    return {
        "status": "ok",
        "source": {
            "title": result["title"],
            "url": result["url"],
            "source_name": result["source_name"],
            "domain": result["domain"],
            "source_tier": result["source_tier"],
        },
        "evidence": evidence,
        "evidence_chars": len(evidence),
    }


def search_external_regulations(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search controlled external regulatory sources.

    V6.1 intentionally uses the official JDIH BPK search endpoint only.
    Results are metadata/snippets; no content is inserted into the RAG.
    """
    query = str(query or "").strip()
    max_results = max(1, min(int(max_results), 10))

    if not query:
        return []

    search_url = _build_bpk_search_url(query)
    logger.info(
        "External regulation search started: query_chars=%d max_results=%d url=%s",
        len(query),
        max_results,
        search_url,
    )

    try:
        html = _fetch(search_url)
    except HTTPError as exc:
        logger.warning(
            "External regulation search HTTP error: code=%s reason=%s url=%s",
            exc.code,
            exc.reason,
            search_url,
        )
        return []
    except URLError as exc:
        logger.warning(
            "External regulation search URL error: reason=%s url=%s",
            exc.reason,
            search_url,
        )
        return []
    except (TimeoutError, ValueError) as exc:
        logger.warning(
            "External regulation search error: type=%s detail=%s url=%s",
            type(exc).__name__,
            exc,
            search_url,
        )
        return []

    if not html:
        logger.warning(
            "External regulation search returned empty HTML: url=%s", search_url)
        return []

    title = _extract_title(html)
    text = _strip_html(html)

    # V6.1 keeps extraction deliberately conservative. If the page structure
    # changes, returning a controlled search-page result is safer than
    # fabricating individual regulations.
    domain = _normalize_domain(search_url)
    source = OFFICIAL_DOMAINS.get(domain)

    if not source:
        return []

    snippet = text[:1000]
    if not snippet:
        logger.warning(
            "External regulation search produced empty text: title=%s url=%s",
            title,
            search_url,
        )
        return []

    logger.info(
        "External regulation search produced search-page evidence: title=%s snippet_chars=%d",
        title,
        len(snippet),
    )

    return [
        {
            "title": title or f"Hasil pencarian regulasi: {query}",
            "url": search_url,
            "snippet": snippet,
            "source_name": source["source_name"],
            "domain": domain,
            "source_tier": source["source_tier"],
        }
    ][:max_results]


def validate_external_source(result: Dict[str, Any]) -> bool:
    """
    Validate an external research result before it can become evidence.

    V6.2 validation is deliberately conservative:
    - result must be a dict
    - URL must use HTTPS
    - hostname must be an allowlisted official domain
    - source metadata must match the allowlist
    - title and snippet must be non-empty
    """
    if not isinstance(result, dict):
        return False

    url = str(result.get("url") or "").strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() != "https":
        return False

    domain = _normalize_domain(url)

    source = OFFICIAL_DOMAINS.get(domain)
    if not source:
        return False

    if result.get("source_tier") != source["source_tier"]:
        return False

    if result.get("source_name") != source["source_name"]:
        return False

    if not str(result.get("title") or "").strip():
        return False

    if not str(result.get("snippet") or "").strip():
        return False

    return True


def validate_external_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only validated external research results.

    Invalid results are discarded rather than passed downstream.
    """
    if not isinstance(results, list):
        return []

    return [
        result
        for result in results
        if validate_external_source(result)
    ]
