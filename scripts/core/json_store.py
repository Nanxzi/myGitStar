import hashlib
import json
import os
from typing import Any, Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


# ============================================================================
# Schema constants
# ============================================================================
# Each entry in repo_summaries.json now carries metadata alongside the
# content fields, so subsequent runs can do incremental updates without
# re-calling the LLM for repos whose upstream description has not changed.
SUMMARY_CONTENT_FIELDS = (
    "Repository Name",
    "Repository URL",
    "Brief Introduction",
    "Innovations",
    "Basic Usage",
    "Summary",
)
SUMMARY_META_FIELDS = (
    "last_summarized_at",       # ISO8601 UTC timestamp of last successful summary
    "description_hash",         # sha256 of (full_name + description) at summarize time
    "summary_model",            # model identifier used to produce the summary
    "summary_source",           # "ai" | "reused" | "backfill" | "manual"
    "summary_attempts",         # number of LLM attempts that produced this entry
)


def compute_description_hash(full_name: str, description: str) -> str:
    """Stable hash of (full_name + description). Used to detect upstream changes.

    We include full_name so a repo that is forked/renamed still counts as
    'changed' even if its description happens to collide.
    """
    payload = f"{str(full_name or '').strip()}\x00{str(description or '').strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def get_summary_json_path(language: str) -> str:
    if language == "zh":
        return os.path.join(REPO_ROOT, "repo_summaries_zh.json")
    return os.path.join(REPO_ROOT, "repo_summaries.json")


def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json_atomic(data: dict, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def make_metadata(
    full_name: str,
    description: str,
    model: str,
    source: str,
    attempts: int = 1,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the metadata dict to embed inside a summary entry.

    This is the single place that knows how a summary is timestamped / hashed
    so the format stays consistent across summarize + classify paths.
    """
    from datetime import datetime, timezone
    return {
        "last_summarized_at": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description_hash": compute_description_hash(full_name, description),
        "summary_model": model or "unknown",
        "summary_source": source,
        "summary_attempts": int(attempts) if attempts else 1,
    }


def is_entry_fresh(
    entry: Optional[Dict[str, Any]],
    full_name: str,
    description: str,
    refresh_after_days: int = 0,
) -> bool:
    """Return True if the entry's summary is still considered fresh.

    "Fresh" means:
      - has the 4 user-visible content fields populated (Repository URL is
        always re-derivable from full_name, so we don't require it), AND
      - description_hash matches the current upstream description, AND
      - if refresh_after_days > 0, last_summarized_at is within that window.

    refresh_after_days=0 means "never auto-refresh even if old" - we still
    invalidate when the description changes. This is the default for the
    GitHub Actions workflow since we want to minimise API calls.
    """
    if not isinstance(entry, dict):
        return False

    required_fields = (
        "Repository Name",
        "Brief Introduction",
        "Innovations",
        "Summary",
    )
    for f in required_fields:
        v = entry.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False

    # `Basic Usage` is allowed to be empty / "Not specified." - some repos
    # genuinely have no usage snippet (e.g. libraries without examples).
    basic = entry.get("Basic Usage")
    if basic is not None and isinstance(basic, str) and not basic.strip():
        pass  # ok

    current_hash = compute_description_hash(full_name, description)
    meta = entry.get("__meta__") if isinstance(entry.get("__meta__"), dict) else {}
    stored_hash = meta.get("description_hash") or entry.get("description_hash")
    if stored_hash and stored_hash != current_hash:
        return False

    if refresh_after_days and refresh_after_days > 0:
        ts = entry.get("__meta__", {}).get("last_summarized_at") if isinstance(entry.get("__meta__"), dict) else None
        if not ts:
            return False
        try:
            from datetime import datetime, timezone, timedelta
            last = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last > timedelta(days=refresh_after_days):
                return False
        except Exception:
            return False

    return True


def normalize_json_store(data: Any) -> Dict[str, Dict]:
    """Normalize JSON store to {full_name: entry_dict}.

    Supports:
    1) full_name -> entry dict (current format)
    2) category -> [entry dict] (extracted from README)
    3) list of entry dicts
    """
    normalized: Dict[str, Dict] = {}
    if not data:
        return normalized

    def _add_entry(full_name: str, entry: Dict) -> None:
        if not full_name:
            return
        normalized[full_name] = entry

    if isinstance(data, dict):
        # category -> [entry dict]
        if all(isinstance(v, list) for v in data.values()):
            for _, items in data.items():
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    full_name = item.get("full_name") or item.get("Repository Name") or item.get("repo")
                    if not full_name:
                        continue
                    summary = item.get("summary") or item.get("Summary") or ""
                    normalized_entry = dict(item)
                    normalized_entry["summary"] = summary
                    normalized_entry["full_name"] = full_name
                    _add_entry(full_name, normalized_entry)
            return normalized

        # full_name -> entry dict or summary string
        for key, value in data.items():
            if isinstance(value, dict):
                full_name = value.get("full_name") or value.get("Repository Name") or key
                summary = value.get("summary") or value.get("Summary") or ""
                normalized_entry = dict(value)
                normalized_entry["summary"] = summary
                normalized_entry["full_name"] = full_name
                _add_entry(full_name, normalized_entry)
            elif isinstance(value, str):
                normalized_entry = {"full_name": key, "summary": value}
                _add_entry(key, normalized_entry)
        return normalized

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            full_name = item.get("full_name") or item.get("Repository Name") or item.get("repo")
            if not full_name:
                continue
            summary = item.get("summary") or item.get("Summary") or ""
            normalized_entry = dict(item)
            normalized_entry["summary"] = summary
            normalized_entry["full_name"] = full_name
            _add_entry(full_name, normalized_entry)
    return normalized


def build_summary_index(json_store: Dict[str, Dict]) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
    for full_name, entry in (json_store or {}).items():
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary") or entry.get("Summary") or ""
        if summary:
            summaries[full_name] = str(summary)
    return summaries


def get_summary_from_json(json_store: Dict[str, Dict], full_name: str) -> str:
    if not json_store or not full_name:
        return ""
    entry = json_store.get(full_name)
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("summary") or entry.get("Summary") or "")


def merge_summary_store(existing_store: Dict[str, Dict], updates: Dict[str, Dict]) -> Dict[str, Dict]:
    merged = dict(existing_store or {})
    for key, value in (updates or {}).items():
        if not key:
            continue
        merged[key] = value
    return merged


def load_summary_store(json_path: str) -> Dict[str, Dict]:
    raw = load_json(json_path)
    return normalize_json_store(raw)
