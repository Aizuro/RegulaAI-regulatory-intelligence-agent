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
import re


# Official / trusted regulatory domains. Keep this allowlist conservative.
OFFICIAL_DOMAINS = {
    "peraturan.bpk.go.id": {
        "source_name": "JDIH BPK",
        "source_tier": 1,
    },
    "jdihn.go.id": {
        "source_name": "JDIHN",
        "source_tier": 1,
    },
    "jdih.kemenkeu.go.id": {
        "source_name": "JDIH Kementerian Keuangan",
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


def _build_jdihn_search_url(query: str) -> str:
    """Build a controlled JDIHN search URL using its public search page."""
    return (
        "https://jdihn.go.id/pencarian?instansi=&jenis="
        f"&keyword={quote(query)}&nomor=&status=&tahun="
    )


def _build_kemenkeu_perpres_url(number: str, year: str) -> str:
    """Build the canonical JDIH Kemenkeu page for a Perpres."""
    return f"https://jdih.kemenkeu.go.id/dok/perpres-{number}-tahun-{year}/overview"


def _extract_perpres_reference(query: str):
    """Extract an explicit Perpres number/year from a user query."""
    match = re.search(
        r"Peraturan\s+Presiden\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1), match.group(2)


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


    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return ""
        body = response.read().decode("utf-8", errors="replace")
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
        return {
            "status": "fetch_failed",
            "source": result,
            "evidence": "",
        }
    except URLError as exc:
        return {
            "status": "fetch_failed",
            "source": result,
            "evidence": "",
        }
    except (TimeoutError, ValueError) as exc:
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

    Search official Tier-1 regulatory sources in priority order. For an
    explicit Perpres reference, use the canonical JDIH Kementerian Keuangan
    page first; BPK and JDIHN remain fallbacks for broader searches. Results
    are metadata/snippets; no content is inserted into the RAG.
    """
    query = str(query or "").strip()
    max_results = max(1, min(int(max_results), 10))

    if not query:
        return []

    # Prefer a deterministic official page when the query explicitly names
    # a Perpres. This avoids relying on search-page endpoints that may be
    # blocked by the cloud deployment environment.
    perpres_reference = _extract_perpres_reference(query)
    if perpres_reference:
        number, year = perpres_reference
        search_targets = [
            (
                "JDIH Kementerian Keuangan",
                _build_kemenkeu_perpres_url(number, year),
            ),
            ("JDIH BPK", _build_bpk_search_url(query)),
            ("JDIHN", _build_jdihn_search_url(query)),
        ]
    else:
        search_targets = [
            ("JDIH BPK", _build_bpk_search_url(query)),
            ("JDIHN", _build_jdihn_search_url(query)),
        ]

    for source_label, search_url in search_targets:

        try:
            html = _fetch(search_url)
        except HTTPError as exc:
            continue
        except URLError as exc:
            continue
        except (TimeoutError, ValueError) as exc:
            continue

        if not html:
            continue

        title = _extract_title(html)
        text = _strip_html(html)

        # Keep extraction deliberately conservative. Returning a controlled
        # search-page result is safer than fabricating individual regulations.
        domain = _normalize_domain(search_url)
        source = OFFICIAL_DOMAINS.get(domain)

        if not source:
            continue

        snippet = text[:1000]
        if not snippet:
            continue


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

    return []


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
