"""V7 regulatory monitoring utilities.

V7.1 defines the API monitoring contract. V7.2 adds deterministic source
normalization, fingerprinting, and snapshot diffing for later automation.
AI relevance analysis and n8n orchestration remain outside this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List

MONITORING_EVENTS = {
    "new_regulation",
    "updated_regulation",
    "removed_regulation",
}


def build_regulation_key(
    document_type: str,
    number: str,
    year: str | int,
) -> str:
    """Create a stable identifier for change-detection/state tracking."""
    normalized_type = re.sub(r"\s+", "-", document_type.strip().upper())
    normalized_number = str(number).strip()
    normalized_year = str(year).strip()
    return f"{normalized_type}-{normalized_number}-{normalized_year}"


def normalize_monitoring_event(
    regulation: Dict[str, Any],
    event: str,
) -> Dict[str, Any]:
    """Normalize a validated monitoring payload into an internal contract."""
    if event not in MONITORING_EVENTS:
        raise ValueError(f"Unsupported monitoring event: {event}")

    return {
        "regulation_key": build_regulation_key(
            regulation["type"], regulation["number"], regulation["year"]
        ),
        "event": event,
        "regulation": {
            "type": regulation["type"].strip(),
            "number": regulation["number"].strip(),
            "year": str(regulation["year"]).strip(),
            "title": regulation["title"].strip(),
            "url": str(regulation["url"]).strip(),
            "source": regulation["source"].strip(),
            "published_date": (
                regulation["published_date"].strip()
                if regulation.get("published_date")
                else None
            ),
        },
    }


def build_regulation_fingerprint(regulation: Dict[str, Any]) -> str:
    """Build a deterministic SHA-256 fingerprint for change detection."""
    normalized = normalize_regulation(regulation)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_regulation(regulation: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical source representation used for state comparison."""
    required = ("type", "number", "year", "title", "url", "source")
    missing = [field for field in required if not str(regulation.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Missing required regulation fields: {', '.join(missing)}")

    return {
        "type": str(regulation["type"]).strip(),
        "number": str(regulation["number"]).strip(),
        "year": str(regulation["year"]).strip(),
        "title": str(regulation["title"]).strip(),
        "url": str(regulation["url"]).strip(),
        "source": str(regulation["source"]).strip(),
        "published_date": (
            str(regulation["published_date"]).strip()
            if regulation.get("published_date")
            else None
        ),
    }


def detect_changes(
    previous: Iterable[Dict[str, Any]],
    current: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare two regulation snapshots and return deterministic change events.

    Each event contains ``regulation_key``, ``event``, ``regulation`` and
    ``fingerprint``. Duplicate keys in either snapshot are rejected because
    they make change detection ambiguous.
    """
    previous_map = _snapshot_map(previous)
    current_map = _snapshot_map(current)
    events: List[Dict[str, Any]] = []

    for key in sorted(current_map):
        current_item = current_map[key]
        current_fingerprint = build_regulation_fingerprint(current_item)
        previous_item = previous_map.get(key)

        if previous_item is None:
            events.append(_change_event(current_item, "new_regulation", current_fingerprint))
        elif build_regulation_fingerprint(previous_item) != current_fingerprint:
            events.append(
                _change_event(
                    current_item,
                    "updated_regulation",
                    current_fingerprint,
                    previous_regulation=previous_item,
                )
            )

    for key in sorted(set(previous_map) - set(current_map)):
        previous_item = previous_map[key]
        events.append(
            _change_event(
                previous_item,
                "removed_regulation",
                build_regulation_fingerprint(previous_item),
            )
        )

    return events


def _snapshot_map(snapshot: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for raw_regulation in snapshot:
        regulation = normalize_regulation(raw_regulation)
        key = build_regulation_key(
            regulation["type"], regulation["number"], regulation["year"]
        )
        if key in result:
            raise ValueError(f"Duplicate regulation_key in snapshot: {key}")
        result[key] = regulation
    return result


def _change_event(
    regulation: Dict[str, Any],
    event: str,
    fingerprint: str,
    previous_regulation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = {
        "regulation_key": build_regulation_key(
            regulation["type"], regulation["number"], regulation["year"]
        ),
        "event": event,
        "regulation": regulation,
        "fingerprint": fingerprint,
    }

    if previous_regulation is not None:
        result["previous_regulation"] = previous_regulation

    return result


def build_monitoring_contract(
    regulation: Dict[str, Any],
    event: str,
) -> Dict[str, Any]:
    """Return the V7 deterministic API response; AI analysis is not performed."""
    normalized = normalize_monitoring_event(regulation, event)

    return {
        "status": "ok",
        "relevance": "unknown",
        "summary": "",
        "key_points": [],
        "should_notify": False,
        "regulation_key": normalized["regulation_key"],
        "event": normalized["event"],
    }
