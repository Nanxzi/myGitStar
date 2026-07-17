import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.config import (
    load_config,
    get_int_config,
    get_float_config,
    env_truthy,
    normalize_update_mode,
    resolve_update_mode,
    get_model_choice,
)
from scripts.core.secrets import load_api_keys
from scripts.core.throttle import SimpleThrottle
from scripts.classification import (
    init_classify_helpers,
    build_taxonomy_prompt,
    build_classification_prompt,
    _resolve_min_repos_per_category,
    _compute_effective_max_categories,
    _sample_repos_for_taxonomy,
)

# NOTE:
# - This script reuses the same config + env secret strategy as scripts/summarize_stars.py
# - It classifies repos by CONTENT (based mainly on repository description), not by language.

config = load_config()


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


GITHUB_TOKEN, OPENROUTER_API_KEY, GEMINI_API_KEY, _ = load_api_keys(config)

MODEL_CHOICE = get_model_choice(config)

DEFAULT_COPILOT_MODEL = config.get("default_copilot_model", "openai/gpt-4o-mini")
DEFAULT_OPENROUTER_MODEL = config.get("default_openrouter_model")
DEFAULT_GEMINI_MODEL = config.get("default_gemini_model", "gemini-2.0-flash")


update_mode = resolve_update_mode(config)

REQUEST_TIMEOUT = float(os.environ.get("MYGITSTAR_REQUEST_TIMEOUT", get_float_config(config, "request_timeout", 30.0)))
REQUEST_RETRY_DELAY = float(os.environ.get("MYGITSTAR_REQUEST_RETRY_DELAY", get_float_config(config, "request_retry_delay", 2.0)))
RETRY_ATTEMPTS = int(os.environ.get("MYGITSTAR_RETRY_ATTEMPTS", get_int_config(config, "retry_attempts", 1)))
RATE_LIMIT_DELAY = float(os.environ.get("MYGITSTAR_RATE_LIMIT_DELAY", get_float_config(config, "rate_limit_delay", 3.0)))
GLOBAL_QPS = float(os.environ.get("MYGITSTAR_GLOBAL_QPS", get_float_config(config, "global_qps", 0.5)))

FALLBACK_ON_429 = env_truthy("MYGITSTAR_FALLBACK_ON_429")

DEBUG_API = env_truthy("DEBUG_API")

init_classify_helpers(config)


class RateLimitAbort(RuntimeError):
    """Raised when too many consecutive 429 responses occur."""


MAX_CONSECUTIVE_429 = int(os.environ.get("MYGITSTAR_MAX_CONSECUTIVE_429", "5") or "5")
_CONSECUTIVE_429 = 0


def _reset_consecutive_429() -> None:
    global _CONSECUTIVE_429
    _CONSECUTIVE_429 = 0


def _note_429_and_maybe_abort(backend: str) -> None:
    global _CONSECUTIVE_429
    _CONSECUTIVE_429 += 1
    if MAX_CONSECUTIVE_429 > 0 and _CONSECUTIVE_429 >= MAX_CONSECUTIVE_429:
        raise RateLimitAbort(
            f"Too many consecutive 429 rate-limit responses (count={_CONSECUTIVE_429}, max={MAX_CONSECUTIVE_429}) from {backend}. Aborting."
        )


THROTTLE = SimpleThrottle(GLOBAL_QPS)


API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def _safe_print_debug(msg: str) -> None:
    if DEBUG_API:
        print(msg)


def make_api_request(url: str, headers: Dict[str, str], data: Dict[str, Any], retries: int, retry_delay: float) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            THROTTLE.wait()
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=REQUEST_TIMEOUT)
            _safe_print_debug(f"[API] POST {url} -> {resp.status_code}")
            if DEBUG_API:
                _safe_print_debug(f"[API] Response text: {resp.text[:2000]}")


            if resp.status_code == 429:
                retry_after = None
                try:
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = int(ra)
                except Exception:
                    retry_after = None

                # 跳过超长等待的批次
                if retry_after is not None and retry_after > 600:
                    print(f"[429] Retry-After={retry_after}s 超过 600s，跳过该批次！")
                    return {"error": {"code": 429, "message": f"Too Many Requests, Retry-After={retry_after}s, skipped batch."}, "status_code": 429}

                if attempt < retries - 1:
                    wait = retry_after if retry_after else (retry_delay * (2 ** attempt) + random.uniform(0, 1))
                    print(f"[429] Rate limited, sleep {wait:.1f}s then retry ({attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue
                return {"error": {"code": 429, "message": "Too Many Requests"}, "status_code": 429}

            # For non-2xx, return a structured error payload instead of raising.
            if not (200 <= int(resp.status_code) < 300):
                body_preview = ""
                try:
                    body_preview = (resp.text or "").strip()[:2000]
                except Exception:
                    body_preview = ""
                return {
                    "error": {
                        "code": int(resp.status_code),
                        "message": f"HTTP {resp.status_code}",
                        "body_preview": body_preview,
                    },
                    "status_code": int(resp.status_code),
                }

            try:
                return resp.json()
            except Exception:
                return {"text": resp.text}
        except requests.HTTPError as e:
            if attempt < retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"[HTTPError] {e}; sleep {wait:.1f}s then retry")
                time.sleep(wait)
                continue
            print(f"[HTTPError] final: {e}")
            return {"error": {"code": "http_error", "message": str(e)}, "status_code": "http_error"}
        except Exception as e:
            if attempt < retries - 1:
                wait = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[Error] {e}; sleep {wait:.1f}s then retry")
                time.sleep(wait)
                continue
            print(f"[Error] final: {e}")
            return {"error": {"code": "exception", "message": str(e)}, "status_code": "exception"}

    return {"error": {"code": "unknown", "message": "request failed"}, "status_code": "unknown"}


def _raise_if_error_response(resp: Optional[Dict[str, Any]], backend: str) -> None:
    if not resp or not isinstance(resp, dict):
        raise RuntimeError(f"{backend} request failed: empty response")
    err = resp.get("error")
    if not err:
        _reset_consecutive_429()
        return
    try:
        code = err.get("code")

        # Track consecutive 429 responses and abort if it persists.
        if str(code) == "429" or resp.get("status_code") == 429:
            _note_429_and_maybe_abort(backend)
        else:
            _reset_consecutive_429()

        msg = err.get("message") or ""
        body_preview = err.get("body_preview") or ""
        details = f"{msg}".strip()
        if body_preview:
            details = (details + "\n" if details else "") + f"body_preview: {body_preview}"
        raise RuntimeError(f"{backend} request failed (code={code})\n{details}".rstrip())
    except RateLimitAbort:
        raise
    except Exception as e:
        raise RuntimeError(f"{backend} request failed (unparseable error payload): {e}")


def _extract_json_from_text(text: str) -> Any:
    """Best-effort JSON extraction (models sometimes wrap JSON in markdown fences)."""
    if text is None:
        raise ValueError("empty response")
    s = str(text).strip()

    # Strip markdown fences
    s = re.sub(r"^```(json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)

    # Direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Find first JSON object/array
    obj_start = s.find("{")
    arr_start = s.find("[")
    start = -1
    if obj_start != -1 and arr_start != -1:
        start = min(obj_start, arr_start)
    else:
        start = obj_start if obj_start != -1 else arr_start

    if start == -1:
        raise ValueError("no json found")

    tail = s[start:]
    # Greedy attempt: try to parse progressively shorter tails
    for end in range(len(tail), 1, -1):
        candidate = tail[:end]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("failed to parse json")


def _get_llm_text(response: Optional[Dict[str, Any]]) -> str:
    if not response:
        return ""
    if "choices" in response:
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and "content" in msg:
                return str(msg.get("content") or "").strip()
            if "content" in choices[0]:
                return str(choices[0].get("content") or "").strip()
    if "text" in response:
        return str(response.get("text") or "").strip()
    return ""


def call_llm(prompt: str) -> str:
    """Call the selected LLM backend with a user prompt. Returns raw text."""
    model_choice = get_model_choice(config)

    if model_choice == "copilot":
        if not GITHUB_TOKEN:
            raise RuntimeError("Missing STARRED_GITHUB_TOKEN (required for copilot backend).")
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", DEFAULT_COPILOT_MODEL) or "openai/gpt-4o-mini"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.2,
        }
        resp = make_api_request(API_ENDPOINTS["copilot"], headers, data, retries=RETRY_ATTEMPTS, retry_delay=REQUEST_RETRY_DELAY)

        # Optional fallback when Copilot is rate-limited (HTTP 429).
        if (
            FALLBACK_ON_429
            and isinstance(resp, dict)
            and isinstance(resp.get("error"), dict)
            and resp.get("error", {}).get("code") == 429
        ):
            print("[WARN] Copilot rate-limited (429). Trying fallback backend...")
            if OPENROUTER_API_KEY:
                # Call OpenRouter directly (avoid recursion + static globals)
                model_name2 = DEFAULT_OPENROUTER_MODEL or "openai/gpt-4o-mini"
                headers2 = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                data2 = {"model": model_name2, "messages": [{"role": "user", "content": prompt}]}
                resp2 = make_api_request(API_ENDPOINTS["openrouter"], headers2, data2, retries=RETRY_ATTEMPTS, retry_delay=REQUEST_RETRY_DELAY)
                _raise_if_error_response(resp2, backend="OpenRouter")
                text2 = _get_llm_text(resp2)
                if not str(text2).strip():
                    raise RuntimeError("OpenRouter returned empty content")
                return text2

            if GEMINI_API_KEY:
                # Call Gemini directly
                model_name2 = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or "gemini-2.0-flash"
                model_path2 = str(model_name2).strip()
                if model_path2.startswith("models/"):
                    model_path2 = model_path2[len("models/"):]

                api_url2 = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path2}:generateContent"
                request_url2 = f"{api_url2}?key={GEMINI_API_KEY}"
                headers2 = {
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "GitHub Star Content Classifier/1.0",
                    "X-Goog-Api-Key": GEMINI_API_KEY,
                }
                temperature2 = config.get("gemini_temperature", 0.2)
                max_output_tokens2 = int(config.get("gemini_max_output_tokens", 2000))
                payload2 = {
                    "generationConfig": {
                        "temperature": temperature2,
                        "maxOutputTokens": max_output_tokens2,
                        "topP": 0.8,
                        "topK": 40,
                    },
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                }
                resp2 = make_api_request(
                    request_url2,
                    headers2,
                    payload2,
                    retries=get_int_config(config, "gemini_retry_attempts", RETRY_ATTEMPTS),
                    retry_delay=get_float_config(config, "gemini_retry_delay", REQUEST_RETRY_DELAY),
                )
                _raise_if_error_response(resp2, backend="Gemini")
                if not isinstance(resp2, dict):
                    raise RuntimeError("Gemini request failed: invalid response type")
                text2 = ""
                try:
                    candidates = resp2.get("candidates") or []
                    if candidates and isinstance(candidates[0], dict):
                        content = candidates[0].get("content") or {}
                        parts = content.get("parts") or []
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                text2 += str(part.get("text") or "")
                except Exception:
                    pass
                text2 = text2.strip()
                if not text2:
                    raise RuntimeError("Gemini returned empty content")
                return text2

        _raise_if_error_response(resp, backend="Copilot")
        text = _get_llm_text(resp)
        if not str(text).strip():
            raise RuntimeError("Copilot returned empty content")
        return text

    if model_choice == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY (required for openrouter backend).")
        model_name = DEFAULT_OPENROUTER_MODEL or "openai/gpt-4o-mini"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        resp = make_api_request(API_ENDPOINTS["openrouter"], headers, data, retries=RETRY_ATTEMPTS, retry_delay=REQUEST_RETRY_DELAY)
        _raise_if_error_response(resp, backend="OpenRouter")
        text = _get_llm_text(resp)
        if not str(text).strip():
            raise RuntimeError("OpenRouter returned empty content")
        return text

    if model_choice == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY (required for gemini backend).")

        model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or "gemini-2.0-flash"
        model_path = str(model_name).strip()
        if model_path.startswith("models/"):
            model_path = model_path[len("models/"):]

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        request_url = f"{api_url}?key={GEMINI_API_KEY}"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GitHub Star Content Classifier/1.0",
            "X-Goog-Api-Key": GEMINI_API_KEY,
        }
        temperature = config.get("gemini_temperature", 0.2)
        max_output_tokens = int(config.get("gemini_max_output_tokens", 2000))
        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.8,
                "topK": 40,
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

        resp = make_api_request(
            request_url,
            headers,
            payload,
            retries=get_int_config(config, "gemini_retry_attempts", RETRY_ATTEMPTS),
            retry_delay=get_float_config(config, "gemini_retry_delay", REQUEST_RETRY_DELAY),
        )
        _raise_if_error_response(resp, backend="Gemini")

        if not isinstance(resp, dict):
            raise RuntimeError("Gemini request failed: invalid response type")

        # Gemini: candidates[0].content.parts[].text
        text = ""
        try:
            candidates = resp.get("candidates") or []
            if candidates and isinstance(candidates[0], dict):
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        text += str(part.get("text") or "")
        except Exception:
            pass
        text = text.strip()
        if not text:
            raise RuntimeError("Gemini returned empty content")
        return text

    if model_choice == "lmstudio":
        from scripts.ai.lmstudio_client import lmstudio_call_llm
        lmstudio_model = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")
        return lmstudio_call_llm(prompt, model=lmstudio_model)

    if model_choice == "ollama":
        from scripts.ai.ollama_client import ollama_call_llm
        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
        return ollama_call_llm(prompt, model=ollama_model)

    raise RuntimeError(f"Unsupported model_choice: {model_choice}")


def call_llm_json(prompt: str, attempts: int = 2) -> Any:
    """Call LLM and parse JSON, retrying once if the model returns non-JSON or truncated JSON."""
    last_error: Optional[Exception] = None
    for i in range(1, max(1, attempts) + 1):
        p = prompt
        if i > 1:
            p = (
                prompt
                + "\n\nIMPORTANT: Return STRICT JSON only. Do not include markdown fences, explanations, or extra text."
            )
        try:
            text = call_llm(p)
            return _extract_json_from_text(text)
        except RateLimitAbort:
            raise
        except Exception as e:
            last_error = e
            if i < attempts:
                time.sleep(1.0)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Failed to get JSON from LLM")


def get_github_username() -> str:
    gh = config.get("github_username") if isinstance(config, dict) else None
    if gh == "0" or gh == 0:
        actor = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
        if not actor:
            raise RuntimeError("github_username is 0 but GITHUB_ACTOR/GITHUB_USERNAME is not set.")
        return actor
    if not gh:
        raise RuntimeError("Missing config.github_username")
    return str(gh)


def get_current_account_optional() -> str:
    """Best-effort current account display.

    Never raises: returns empty string if it cannot be determined.
    """
    try:
        gh = config.get("github_username") if isinstance(config, dict) else None
        if gh and gh != "0" and gh != 0:
            return str(gh).strip()

        actor = (os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME") or "").strip()
        if actor:
            return actor
    except Exception:
        return ""

    return ""


def get_starred_repos(max_repos: Optional[int] = None) -> List[Dict[str, Any]]:
    if not GITHUB_TOKEN:
        raise RuntimeError("Missing STARRED_GITHUB_TOKEN")

    username = get_github_username()
    print(f"Fetching starred repos for: {username}")

    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"https://api.github.com/users/{username}/starred?per_page={per_page}&page={page}"
        THROTTLE.wait()
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
        if not data:
            break
        repos.extend(data)
        print(f"  got {len(repos)} repos... (page {page})")
        page += 1
        time.sleep(1)

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            repos = repos[:max_repos]
            break

    return repos


_KEYWORD_MAP = {
    "agent": "AI Agent Autonomous",
    "agents": "AI Agent Autonomous",
    "autonomous": "Autonomous Agent",
    "auto": "Automation",
    "gpt": "LLM GPT Language Model",
    "llm": "LLM Language Model",
    "claude": "LLM Claude",
    "chat": "Chatbot Assistant",
    "bot": "Bot Assistant",
    "assistant": "AI Assistant",
    "research": "Research Automation",
    "lab": "Research Laboratory",
    "laboratory": "Research Laboratory",
    "write": "Writing Content Creation",
    "writing": "Writing Content Creation",
    "novel": "Novel Writing Content Creation",
    "video": "Video Content Creation",
    "image": "Image Vision",
    "vision": "Computer Vision",
    "cv": "Computer Vision",
    "audio": "Audio Speech",
    "speech": "Speech Audio",
    "code": "Coding Developer",
    "coding": "Coding Developer",
    "dev": "Developer Tools",
    "skill": "Agent Skills",
    "skills": "Agent Skills",
    "mcp": "MCP Tool Integration",
    "tool": "Tool Utility",
    "framework": "Framework Platform",
    "open": "Open Source",
    "awesome": "Awesome List Collection",
    "learn": "Learning Education",
    "tutorial": "Learning Tutorial",
    "beginner": "Beginner Learning",
    "mini": "Mini Lightweight",
    "small": "Small Lightweight",
    "fast": "Fast Performance",
    "efficient": "Efficient Optimization",
    "smart": "Smart AI",
    "ai": "Artificial Intelligence",
    "machine": "Machine Learning",
    "ml": "Machine Learning",
    "deep": "Deep Learning",
    "neural": "Neural Network Deep Learning",
    "data": "Data Science",
    "kosmos": "Kosmos AI System",
    "hyperframes": "Hyperframes Video Agent",
    "inkos": "Inkos Novel Writing Agent",
    "multica": "Multica Agent Platform",
    "hyperframes": "Hyperframes Video Generation",
}


def _extract_keywords_from_name(name: str) -> str:
    """Extract meaningful keywords from a repository name for classification context."""
    if not name:
        return ""
    name_lower = name.lower()
    found_keywords: set[str] = set()
    for keyword, context in _KEYWORD_MAP.items():
        if keyword in name_lower:
            for kw in context.split():
                found_keywords.add(kw)
    if found_keywords:
        return ", ".join(sorted(found_keywords))
    return ""


def parse_repos_from_summaries(
    summaries_path: Optional[str] = None,
    max_repos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parse repo list from repo_summaries.json.

    Uses the structured summary data (Brief Introduction, Innovations, etc.)
    as content description for classification.
    """
    if summaries_path is None:
        en_path = os.path.join(REPO_ROOT, "repo_summaries_en.json")
        zh_path = os.path.join(REPO_ROOT, "repo_summaries_zh.json")
        if os.path.exists(en_path):
            summaries_path = en_path
        elif os.path.exists(zh_path):
            summaries_path = zh_path
        else:
            summaries_path = os.path.join(REPO_ROOT, "repo_summaries.json")
    elif not os.path.isabs(summaries_path):
        summaries_path = os.path.join(REPO_ROOT, summaries_path)

    if not os.path.exists(summaries_path):
        raise FileNotFoundError(f"summaries file not found: {summaries_path}")

    with open(summaries_path, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)

    repos: List[Dict[str, Any]] = []
    for idx, (full_name, data) in enumerate(summaries_data.items()):
        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            break

        brief_intro = data.get("Brief Introduction", "")
        innovations = data.get("Innovations", "")
        summary = data.get("Summary", "")

        bi_empty = not brief_intro or brief_intro in ("Not specified.", "")
        in_empty = not innovations or innovations in ("Not specified.", "")
        su_empty = not summary or summary in ("Not specified.", "")
        is_unknown = bi_empty and in_empty and su_empty

        description_parts = []
        if not is_unknown:
            if brief_intro and brief_intro not in ("Not specified.", ""):
                description_parts.append(brief_intro)
            if innovations and innovations not in ("Not specified.", ""):
                description_parts.append(f"创新点: {innovations}")
            if summary and summary not in ("Not specified.", ""):
                description_parts.append(f"总结: {summary}")

        repo_title = full_name.split("/")[-1] if "/" in full_name else full_name
        if is_unknown:
            keywords = _extract_keywords_from_name(repo_title)
            if keywords:
                brief_intro = f"Repository: {repo_title} ({keywords})"
            else:
                brief_intro = f"Repository: {repo_title}"
            description = brief_intro
        else:
            description = " | ".join(description_parts) if description_parts else ""

        description = " | ".join(description_parts) if description_parts and not is_unknown else description

        repos.append(
            {
                "id": idx + 1,
                "full_name": full_name,
                "description": description,
                "brief_intro": brief_intro,
                "innovations": innovations,
                "summary": summary,
                "is_unknown": is_unknown,
                "html_url": data.get("Repository URL", f"https://github.com/{full_name}"),
                "language": data.get("language", ""),
                "stargazers_count": data.get("Stars", data.get("stargazers_count", 0)),
                "forks_count": data.get("Forks", data.get("forks_count", 0)),
                "updated_at": data.get("updated", data.get("updated_at", "")),
                "summary_data": data,
            }
        )

    if not repos:
        raise ValueError(f"No repositories found in summaries file: {summaries_path}")

    return repos


def _clean_inline_md(text: str) -> str:
    s = (text or "").strip()
    # remove trailing markdown double spaces used for line breaks
    s = re.sub(r"\s{2,}$", "", s)
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_repo_stats_or_meta_line(line: str) -> bool:
    """Detect non-description noise lines in generated README blocks.

    This is used ONLY for extracting a clean description for classification.
    We intentionally ignore statistics/metadata so the LLM focuses on repo content.
    """
    s = _clean_inline_md(line)
    if not s:
        return True

    # Markdown fences / separators
    if s.startswith("```"):
        return True
    if s == "---" or s.startswith("---"):
        return True

    # Common generated stats lines
    if s.startswith("**⭐") or s.startswith("**🍴") or s.startswith("**📅"):
        return True
    if "Stars:" in s or "Forks:" in s or "Updated:" in s:
        # Keep it conservative; these are almost always metadata.
        return True

    # Common generated meta labels
    lower = s.lower()
    if lower.startswith("here's the summary") or lower.startswith("here is the summary"):
        return True
    if lower.startswith("repository url") or lower.startswith("**repository url"):
        return True
    if lower.startswith("repository name") or lower.startswith("**repository name"):
        return True

    # Chinese meta labels
    if "仓库url" in s.lower() or "仓库链接" in s:
        return True
    if "仓库名称" in s or "repository name" in lower:
        return True

    # Common generation failure placeholders (not repo content)
    if "生成失败" in s or "rate limit" in lower or "ratelimit" in lower:
        return True
    if re.search(r"\b429\b", s):
        return True

    return False


def _extract_structured_fields_from_block(block: str) -> Dict[str, str]:
    """Extract structured content fields from a generated per-repo markdown block."""
    if not block:
        return {}

    patterns = {
        "brief_intro": [
            r"^2\.\s*\*\*Brief Introduction:\*\*\s*(?P<v>.+?)\s*$",
            r"^2\.\s*\*\*简要介绍：\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*Repository Description:\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*仓库描述：\*\*\s*(?P<v>.+?)\s*$",
        ],
        "innovations": [
            r"^3\.\s*\*\*Innovations:\*\*\s*(?P<v>.+?)\s*$",
            r"^3\.\s*\*\*创新点：\*\*\s*(?P<v>.+?)\s*$",
        ],
        "summary": [
            r"^5\.\s*\*\*Summary:\*\*\s*(?P<v>.+?)\s*$",
            r"^5\.\s*\*\*总结：\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*Summary:\*\*\s*(?P<v>.+?)\s*$",
            r"^\*\*总结：\*\*\s*(?P<v>.+?)\s*$",
        ],
    }

    out: Dict[str, str] = {}
    for key, pats in patterns.items():
        for ep in pats:
            mm = re.search(ep, block, flags=re.MULTILINE)
            if mm:
                val = _clean_inline_md(mm.group("v"))
                if val:
                    out[key] = val
                break

    return out


def _enrich_repos_from_summaries(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge enriched summary fields from repo_summaries.json into repos.

    This ensures that even when loading repos from GitHub API (default mode),
    we still get the rich brief_intro/innovations/summary content for classification.
    """
    summaries_path = os.path.join(REPO_ROOT, "repo_summaries_en.json")
    if not os.path.exists(summaries_path):
        summaries_path = os.path.join(REPO_ROOT, "repo_summaries_zh.json")
    if not os.path.exists(summaries_path):
        summaries_path = os.path.join(REPO_ROOT, "repo_summaries.json")
    if not os.path.exists(summaries_path):
        return repos

    try:
        with open(summaries_path, "r", encoding="utf-8") as f:
            summaries_data = json.load(f)
    except Exception:
        return repos

    summaries_by_name = {str(k).strip(): v for k, v in summaries_data.items()}
    enriched = 0
    for r in repos:
        full_name = str(r.get("full_name") or "").strip()
        data = summaries_by_name.get(full_name)
        if not data:
            continue
        # Only enrich if repo doesn't already have these fields populated
        if not str(r.get("brief_intro") or "").strip():
            bi = data.get("Brief Introduction", "")
            if bi and bi != "Not specified.":
                r["brief_intro"] = bi
                enriched += 1
        if not str(r.get("innovations") or "").strip():
            inn = data.get("Innovations", "")
            if inn and inn != "Not specified.":
                r["innovations"] = inn
        if not str(r.get("summary") or "").strip():
            sm = data.get("Summary", "")
            if sm and sm != "Not specified.":
                r["summary"] = sm
        if not str(r.get("description") or "").strip():
            # Build description from summary fields
            parts = []
            for field_key, label in [
                ("brief_intro", ""),
                ("innovations", "创新点: "),
                ("summary", "总结: "),
            ]:
                v = r.get(field_key, "")
                if v and v != "Not specified.":
                    parts.append(f"{label}{v}" if label else v)
            if parts:
                r["description"] = " | ".join(parts)

    if enriched > 0:
        print(f"[info] Enriched {enriched} repos from existing repo_summaries.json")
    return repos


def _build_repo_content_text(repo: Dict[str, Any]) -> str:
    """Create a stable, content-focused text blob for taxonomy + classification."""
    full_name = _clean_inline_md(str(repo.get("full_name") or ""))
    title = full_name.split("/")[-1] if "/" in full_name else full_name

    parts: List[str] = []
    if title:
        parts.append(title)
    if full_name:
        parts.append(f"GitHub repository: {full_name}")

    # Prefer structured fields when available.
    for k in ("brief_intro", "innovations", "summary"):
        v = _clean_inline_md(str(repo.get(k) or ""))
        if v and not _is_repo_stats_or_meta_line(v):
            parts.append(v)

    # Fallback to extracted description.
    desc = _clean_inline_md(str(repo.get("description") or ""))
    if desc and not _is_repo_stats_or_meta_line(desc):
        parts.append(desc)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: List[str] = []
    for p in parts:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    result = " | ".join(uniq).strip()
    if not result or result.lower() == title.lower():
        result = f"GitHub repository {full_name}"

    return result


def _extract_description_from_block(block: str) -> str:
    """Extract a clean, content-focused description from a repo markdown block."""
    if not block:
        return ""

    extract_patterns = [
        # Preferred: brief introduction / short intro
        r"^2\.\s*\*\*Brief Introduction:\*\*\s*(?P<d>.+?)\s*$",
        r"^2\.\s*\*\*简要介绍：\*\*\s*(?P<d>.+?)\s*$",
        # Also accept direct description labels
        r"^\*\*Repository Description:\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*仓库描述：\*\*\s*(?P<d>.+?)\s*$",
        # Fallback: summary can still represent repo content
        r"^5\.\s*\*\*Summary:\*\*\s*(?P<d>.+?)\s*$",
        r"^5\.\s*\*\*总结：\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*Summary:\*\*\s*(?P<d>.+?)\s*$",
        r"^\*\*总结：\*\*\s*(?P<d>.+?)\s*$",
    ]
    for ep in extract_patterns:
        mm = re.search(ep, block, flags=re.MULTILINE)
        if mm:
            desc = _clean_inline_md(mm.group("d"))
            if desc:
                return desc

    # Last resort: pick first meaningful non-meta line.
    for line in str(block).splitlines():
        s = _clean_inline_md(line)
        if not s:
            continue
        if _is_repo_stats_or_meta_line(s):
            continue

        # Skip numbered meta items like "1. **Repository Name:** ..."
        if re.match(r"^\d+\.\s*\*\*Repository Name:\*\*", s, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d+\.\s*\*\*仓库名称：\*\*", s):
            continue

        # If it's a numbered item, strip the prefix.
        s = re.sub(r"^\d+\.\s*", "", s)
        s = re.sub(r"^[-*]\s+", "", s)
        s = s.strip()
        if s:
            return s

    return ""


def parse_repos_from_readme(readme_path: str, max_repos: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse repo list from an existing generated README.

    Supports headings like:
    - ### 📌 [owner/repo](https://github.com/owner/repo)

    Extracts a short content description from one of:
    - '2. **Brief Introduction:** ...'
    - '2. **简要介绍：** ...'
    - '**Repository Description:** ...'
    - '**仓库描述：** ...'
    """
    if not os.path.isabs(readme_path):
        readme_path = os.path.join(REPO_ROOT, readme_path)

    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README not found: {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by repo heading
    # Example: ### 📌 [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0)
    # Using re.MULTILINE | re.DOTALL and allowing \r for Windows line endings
    pattern = re.compile(
        r"^###\s+📌\s+\[(?P<full_name>[^\]]+)\]\((?P<url>https?://[^\)\s]+)\)\s*[\r]?\n",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))
    repos: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        heading_line = m.group(0)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block = content[start:end]
        block = _trim_repo_block_before_language_section(block)
        block_md = (heading_line + "\n" + block).strip() + "\n"

        full_name = _clean_inline_md(m.group("full_name"))
        url = _clean_inline_md(m.group("url"))

        fields = _extract_structured_fields_from_block(block)
        desc = _extract_description_from_block(block)

        repos.append(
            {
                "id": idx + 1,
                "full_name": full_name,
                "description": desc,
                "html_url": url,
                "block_md": block_md,
                **fields,
            }
        )

        repos[-1]["content_text"] = _build_repo_content_text(repos[-1])

        if max_repos and max_repos > 0 and len(repos) >= max_repos:
            break

    if not repos:
        raise ValueError(
            "No repositories found in README. Expected headings like: ### 📌 [owner/repo](https://github.com/owner/repo)"
        )

    return repos


def _slugify_heading(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "-")
    return s


def _strip_leading_symbols(text: str) -> str:
    """Strip leading emoji/symbol decorations often used in headings."""
    s = (text or "").strip()
    # Remove a run of non-word symbols (emoji/punct) at the start.
    s = re.sub(r"^[^\w\u4e00-\u9fff]+\s*", "", s)
    return s.strip()


_LANGUAGE_CATEGORY_ALIASES = {
    "c#",
    "c++",
    "c",
    "cpp",
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "golang",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "dart",
    "r",
    "matlab",
    "shell",
    "bash",
    "powershell",
    "html",
    "css",
    "vue",
    "react",
    "jupyter notebook",
}


def _looks_like_language_category(name: str) -> bool:
    s = _strip_leading_symbols(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return False
    if s in _LANGUAGE_CATEGORY_ALIASES:
        return True
    # Allow patterns like "C# (something)" or "Python projects"
    head = s.split("(", 1)[0].strip()
    head = head.replace("项目", "").strip()
    if head in _LANGUAGE_CATEGORY_ALIASES:
        return True
    if any(s.startswith(lang + " ") for lang in _LANGUAGE_CATEGORY_ALIASES):
        return True
    return False


def _trim_repo_block_before_language_section(block: str) -> str:
    """Trim accidental language section headings inside a per-repo block.

    In README_lang.md / README_lang_zh.md, language buckets are rendered as level-2 headings like:
    - "## 📝 Jupyter Notebook (Total 3)"
    - "## 📝 Jupyter Notebook（共3个）"

    Our parser splits only by per-repo headings ("### 📌 ..."), so those bucket headings can end up
    inside the previous repo's block. If we carry that block into the content-classified README,
    the language markers will leak into README.md.

    This helper stops copying lines once it detects such a language-bucket heading.
    """
    if not block:
        return ""

    kept: List[str] = []
    for line in str(block).splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[len("## ") :].strip()
            # Strip common count suffixes: (Total N) / （共N个）
            title = re.sub(r"\s*\(\s*total\s*\d+\s*\)\s*$", "", title, flags=re.IGNORECASE)
            title = re.sub(r"（\s*共\s*\d+\s*个\s*）\s*$", "", title)
            title = title.strip()
            if _looks_like_language_category(title):
                break
        kept.append(line)

    return ("\n".join(kept)).rstrip() + "\n"


def _build_category_anchors(taxonomy: "Taxonomy") -> Dict[str, str]:
    """Create stable, unique anchors for categories.

    We use explicit <a id="..."> anchors to guarantee ToC links work,
    independent of GitHub's heading slug algorithm.
    """
    used: set[str] = set()
    anchors: Dict[str, str] = {}
    for c in taxonomy.categories:
        base = _slugify_heading(_strip_leading_symbols(c.get("name", "")))
        if not base:
            base = f"category-{str(c.get('id', '')).strip().lower() or 'x'}"

        anchor = base
        i = 2
        while anchor in used:
            anchor = f"{base}-{i}"
            i += 1
        used.add(anchor)
        anchors[str(c.get("id"))] = anchor
    return anchors


@dataclass
class Taxonomy:
    categories: List[Dict[str, Any]]


def load_existing_categories(path: str) -> Tuple[Optional[Taxonomy], Dict[str, str], Dict[str, str], Optional[str]]:
    """Load existing taxonomy + assignment map from repo_categories.json.

    Returns (taxonomy, assignments_by_full_name, summaries_by_full_name, generated_at).
    The generated_at timestamp lets the caller decide whether the cache is
    still fresh enough to reuse.
    summaries_by_full_name maps full_name -> summary for repos classified as "Other",
    so we can detect summary updates and trigger re-classification.
    """
    if not path or not os.path.exists(path):
        return None, {}, {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, {}, {}, None
    if not isinstance(data, dict):
        return None, {}, {}, None
    taxonomy_raw = data.get("taxonomy") or {}
    categories = taxonomy_raw.get("categories") if isinstance(taxonomy_raw, dict) else None
    if not isinstance(categories, list):
        return None, {}, {}, data.get("generated_at")
    try:
        taxonomy = Taxonomy(categories=[
            {
                "id": str(c.get("id") or "").strip(),
                "name": str(c.get("name") or "").strip(),
                "description": str(c.get("description") or "").strip(),
            }
            for c in categories
            if isinstance(c, dict)
        ])
    except Exception:
        taxonomy = None
    assignments_map: Dict[str, str] = {}
    summaries_map: Dict[str, str] = {}
    assignments = data.get("assignments")
    if isinstance(assignments, list):
        for a in assignments:
            if not isinstance(a, dict):
                continue
            full_name = str(a.get("full_name") or "").strip()
            category_id = str(a.get("category_id") or "").strip()
            if full_name and category_id:
                assignments_map[full_name] = category_id
                # Store summary for Other/Unknown repos to detect updates
                if category_id.lower() in ("other", "unknown"):
                    summary = str(a.get("summary") or "").strip()
                    if summary:
                        summaries_map[full_name] = summary
    return taxonomy, assignments_map, summaries_map, data.get("generated_at")


def _is_taxonomy_cache_fresh(generated_at: Optional[str], max_age_days: int) -> bool:
    """Return True if the cached taxonomy is younger than max_age_days."""
    if not generated_at or not max_age_days or max_age_days <= 0:
        return False
    try:
        from datetime import datetime, timezone, timedelta
        ts = datetime.strptime(generated_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - ts <= timedelta(days=max_age_days)
    except Exception:
        return False


def _should_refresh_taxonomy(
    cached_tax: Optional[Taxonomy],
    cached_assigns: Dict[str, str],
    cached_at: Optional[str],
    current_repos: List[Dict[str, Any]],
    max_age_days: int,
    min_categories: int,
    max_categories: int,
    min_repos_per_category: int,
    new_repo_ratio_threshold: float = 0.3,
) -> Tuple[bool, str]:
    """Decide whether the cached taxonomy should be refreshed.

    Returns (should_refresh, reason).
    """
    if cached_tax is None:
        return True, "no cached taxonomy"

    # Check time freshness
    if not _is_taxonomy_cache_fresh(cached_at, max_age_days):
        return True, f"cache expired (age > {max_age_days} days)"

    # Check category count bounds
    cat_count = len(cached_tax.categories)
    if cat_count < min_categories:
        return True, f"too few categories ({cat_count} < {min_categories})"
    if cat_count > max_categories:
        return True, f"too many categories ({cat_count} > {max_categories})"

    # Check repo collection change
    cached_repo_count = len(cached_assigns)
    current_repo_count = len(current_repos)

    if current_repo_count == 0:
        return False, ""

    # If too many new repos, refresh taxonomy to better fit the new distribution
    new_repo_ratio = max(0, current_repo_count - cached_repo_count) / max(current_repo_count, 1)
    if new_repo_ratio > new_repo_ratio_threshold:
        return True, f"too many new repos ({new_repo_ratio:.0%} > {new_repo_ratio_threshold:.0%})"

    # Check category quality: are any categories too small or too large?
    if min_repos_per_category > 0 and cached_assigns:
        # Count repos per category
        cat_counts: Dict[str, int] = {}
        for cid in cached_assigns.values():
            cat_counts[cid] = cat_counts.get(cid, 0) + 1

        # Check for undersized categories (excluding Other/Unknown)
        undersized = []
        for cat in cached_tax.categories:
            cid = cat["id"]
            if cid.lower() in ("other", "unknown"):
                continue
            count = cat_counts.get(cid, 0)
            if count < min_repos_per_category:
                undersized.append((cid, count))

        if len(undersized) > 2:
            names = [f"{cid}({c})" for cid, c in undersized[:3]]
            return True, f"multiple undersized categories: {', '.join(names)}"

    return False, ""


def _check_taxonomy_quality(
    taxonomy: Taxonomy,
    assignment_map: Dict[Any, str],
    min_repos_per_category: int,
    min_categories: int,
    max_categories: int,
) -> List[str]:
    """Check taxonomy quality and return a list of issues found.

    Issues include:
      - Categories with too few repos (< min_repos_per_category)
      - Category count outside [min_categories, max_categories]
      - Overly dominant category (exceeds balanced threshold based on min_categories)
    """
    issues = []

    # Count repos per category
    cat_counts: Dict[str, int] = {}
    for cid in assignment_map.values():
        cat_counts[cid] = cat_counts.get(cid, 0) + 1

    total = sum(cat_counts.values()) if cat_counts else 1

    # Check category count bounds
    cat_count = len(taxonomy.categories)
    if cat_count < min_categories:
        issues.append(f"Too few categories: {cat_count} < {min_categories}")
    if cat_count > max_categories:
        issues.append(f"Too many categories: {cat_count} > {max_categories}")

    # Check undersized categories
    if min_repos_per_category > 0:
        for cat in taxonomy.categories:
            cid = cat["id"]
            if cid.lower() in ("other", "unknown"):
                continue
            count = cat_counts.get(cid, 0)
            if count < min_repos_per_category:
                issues.append(f"Category '{cat['name']}' has only {count} repos (min={min_repos_per_category})")

    # Check for overly dominant category
    # Balanced threshold: if we have min_categories, no single category should dominate
    # e.g., min_categories=5 means max ~30% per category; min_categories=12 means max ~15%
    # Use a formula: max_ratio = 1.0 / min_categories * 1.5 (with some tolerance)
    max_balanced_ratio = min(0.5, 1.0 / max(min_categories, 1) * 1.5)
    for cat in taxonomy.categories:
        cid = cat["id"]
        if cid.lower() in ("other", "unknown"):
            continue
        count = cat_counts.get(cid, 0)
        if total > 0 and count / total > max_balanced_ratio:
            issues.append(f"Category '{cat['name']}' is overly dominant: {count}/{total} repos ({count/total:.0%} > {max_balanced_ratio:.0%})")

    return issues


def _check_classification_stability(
    reused_count: int,
    total_repos: int,
    unknown_count: int,
    stability_threshold: float,
) -> Tuple[bool, float]:
    """Check if classification results are stable across runs.

    Returns (is_stable, reuse_ratio).
    """
    # Exclude unknown repos from the calculation
    classifiable = total_repos - unknown_count
    if classifiable <= 0:
        return True, 1.0

    reuse_ratio = reused_count / classifiable
    is_stable = reuse_ratio >= stability_threshold
    return is_stable, reuse_ratio


def _is_assignment_still_valid(
    assignment_category_id: str,
    repo: Dict[str, Any],
    taxonomy: Taxonomy,
) -> bool:
    """Decide whether a cached classification should be reused.

    The assignment is considered reusable if:
      * the category id still exists in the current taxonomy, AND
      * the category is not 'Other' or 'Unknown' (these are easy defaults
        the LLM often defaults to, so we'd rather re-evaluate), AND
      * the repo's summary is non-empty (we don't re-classify Unknown repos
        anyway, but be explicit).
    """
    if not assignment_category_id:
        return False
    valid_ids = {c["id"] for c in taxonomy.categories}
    if assignment_category_id not in valid_ids:
        return False
    if assignment_category_id.lower() in ("other", "unknown"):
        return False
    # The repo should have a non-empty summary, otherwise the LLM cannot
    # make a meaningful classification anyway.
    summary = (repo.get("summary") or "").strip()
    brief = (repo.get("brief_intro") or "").strip()
    return bool(summary) or bool(brief)


def _looks_like_language_category(cat: Dict[str, Any]) -> bool:
    name_lower = cat.get("name", "").lower()
    desc_lower = cat.get("description", "").lower()
    text = name_lower + " " + desc_lower

    lang_keywords = [
        "language",
        "programming language",
        "语言",
        "语言实现",
        "interpreter",
        "compiler",
        "runtime",
    ]
    return any(kw in text for kw in lang_keywords)


def _normalize_taxonomy(raw: Any, min_categories: int, max_categories: int) -> Taxonomy:
    if not isinstance(raw, dict):
        raise ValueError("taxonomy JSON must be a dict")

    categories_raw = raw.get("categories")
    if not isinstance(categories_raw, list):
        raise ValueError("taxonomy JSON missing 'categories' list")

    categories = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for c in categories_raw:
        if not isinstance(c, dict):
            continue
        cat_id = str(c.get("id", "")).strip()
        cat_name = str(c.get("name", "")).strip()
        if not cat_id or not cat_name:
            continue
        if cat_id in seen_ids or cat_name.lower() in {n.lower() for n in seen_names}:
            continue
        seen_ids.add(cat_id)
        seen_names.add(cat_name)
        categories.append({
            "id": cat_id,
            "name": cat_name,
            "description": str(c.get("description", "")).strip(),
        })

    filtered = [c for c in categories if not _looks_like_language_category(c)]

    if len(filtered) < min_categories and len(categories) >= min_categories:
        filtered = categories

    if len(filtered) > max_categories:
        filtered = filtered[:max_categories]

    if len(filtered) < min_categories:
        filtered = filtered + [
            {"id": f"CAT{i+1}", "name": f"Category {i+1}", "description": ""}
            for i in range(len(filtered), min_categories)
        ]

    has_other_by_id = any(c["id"].lower() == "other" for c in filtered)
    has_other_by_name = any(c["name"].lower() == "other" for c in filtered)
    if not has_other_by_id and not has_other_by_name:
        filtered.append({"id": "Other", "name": "Other", "description": "Repositories that do not fit into the specified categories."})
    elif has_other_by_name and not has_other_by_id:
        for c in filtered:
            if c["name"].lower() == "other":
                c["id"] = "Other"
                break

    return Taxonomy(categories=filtered)


def _parse_assignments(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw

    if not isinstance(raw, dict):
        return []

    assignments = raw.get("assignments", [])
    if isinstance(assignments, list):
        return assignments

    result = []
    for k, v in raw.items():
        if k == "assignments":
            continue
        if isinstance(v, str):
            result.append({"id": k, "category_id": v})
    return result


def _apply_min_repos_per_category(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    min_repos_per_category: int,
    min_categories: int,
) -> Tuple[Taxonomy, Dict[Any, str]]:
    if min_repos_per_category <= 0:
        return taxonomy, assignment_map

    buckets: Dict[str, List[Any]] = {c["id"]: [] for c in taxonomy.categories}
    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid)
        if cid and cid in buckets:
            buckets[cid].append(rid)

    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])
    unknown_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "unknown"), None)
    if other_id in buckets and len(buckets[other_id]) == 0:
        del buckets[other_id]

    tiny_ids = [cid for cid, rids in buckets.items() if 0 < len(rids) < min_repos_per_category and cid != unknown_id]
    if len(tiny_ids) + len(buckets) <= min_categories:
        tiny_ids = []

    for tid in tiny_ids:
        for rid in buckets[tid]:
            assignment_map[rid] = other_id
        del buckets[tid]

    remaining_cats = [c for c in taxonomy.categories if c["id"] in buckets]
    return Taxonomy(categories=remaining_cats), assignment_map


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def render_markdown(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    cfg: Dict[str, Any],
    model_ch: str,
    username: str = "unknown",
    generated_at: str = None,
    language: str = "en",
) -> str:
    categories = {c["id"]: c for c in taxonomy.categories}
    buckets: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in categories.keys()}
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])

    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid, other_id)
        if cid not in buckets:
            cid = other_id
        buckets[cid].append(r)

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    is_en = (language == "en")

    lines: List[str] = []
    lines.append("<div align=\"center\">\n")
    if is_en:
        lines.append("\n<h1>My GitHub Star Project AI Summary</h1>\n")
        lines.append(f"\n<p><b>Reference Repository:</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n")
        lines.append(f"\n<p><b>Current account:</b> <a href=\"https://github.com/{username}\">{username}</a></p>\n")
        lines.append(f"\n<p><b>Generated on:</b> {generated_at}</p>\n")
        lines.append(f"\n<p><b>AI Model:</b> {model_ch}</p>\n")
        lines.append(f"\n<p><b>Total repositories:</b> {len(repos)}</p>\n")
    else:
        lines.append("\n<h1>我的 GitHub Star 项目AI总结</h1>\n")
        lines.append(f"\n<p><b>参考仓库：</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n")
        lines.append(f"\n<p><b>当前账号：</b> <a href=\"https://github.com/{username}\">{username}</a></p>\n")
        lines.append(f"\n<p><b>生成时间：</b> {generated_at}</p>\n")
        lines.append(f"\n<p><b>AI模型：</b> {model_ch}</p>\n")
        lines.append(f"\n<p><b>总仓库数：</b> {len(repos)} 个</p>\n")
    lines.append("\n</div>\n")
    lines.append("\n---\n")

    lines.append("## Table of Contents\n" if is_en else "## 目录\n")
    sort_by_count = True
    try:
        if isinstance(cfg, dict) and cfg.get("content_sort_categories_by_count") is not None:
            sort_by_count = bool(cfg.get("content_sort_categories_by_count"))
    except Exception:
        sort_by_count = True

    ordered = taxonomy.categories
    if sort_by_count:
        ordered = sorted(
            taxonomy.categories,
            key=lambda c: (
                1 if str(c.get("name", "")).strip().lower() == "other" else 0,
                -len(buckets.get(str(c.get("id")), [])),
                str(c.get("name", "")),
            ),
        )

    for c in ordered:
        anchor = re.sub(r"[^a-z0-9\- ]", "", c["name"].lower()).strip().replace(" ", "-")
        lines.append(f"- [{c['name']}](#{anchor}) ({len(buckets.get(c['id'], []))})\n")
    lines.append("---\n")

    for c in ordered:
        anchor = re.sub(r"[^a-z0-9\- ]", "", c["name"].lower()).strip().replace(" ", "-")
        lines.append(f"<a id=\"{anchor}\"></a>\n")
        lines.append(f"## {c['name']} (Total {len(buckets.get(c['id'], []))})\n\n")
        if c.get("description"):
            lines.append(f"> {c['description']}\n\n")

        bucket = sorted(buckets.get(c["id"], []), key=lambda x: (x.get("full_name") or ""))
        for r in bucket:
            full_name = r.get("full_name") or ""
            url = r.get("html_url") or f"https://github.com/{full_name}"
            stars = r.get("stargazers_count", 0)
            forks = r.get("forks_count", 0)
            updated_at = r.get("updated_at", "")
            if updated_at:
                try:
                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    updated_at_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    updated_at_str = updated_at[:10] if len(updated_at) >= 10 else updated_at
            else:
                updated_at_str = ""

            lines.append(f"### 📌 [{full_name}]({url})\n\n")
            lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **📅 Updated:** {updated_at_str}\n\n")

            repo_summary = r.get("summary_data", {})
            if not isinstance(repo_summary, dict):
                repo_summary = {}

            repo_name = repo_summary.get("Repository Name", full_name)
            brief_intro = repo_summary.get("Brief Introduction", "")
            innovations = repo_summary.get("Innovations", "")
            basic_usage = repo_summary.get("Basic Usage", "")
            summary_text = repo_summary.get("Summary", "")

            lines.append("1. **Repository Name:** " + repo_name + "\n")
            lines.append("2. **Brief Introduction:** " + (brief_intro or "Not specified.") + "\n")
            lines.append("3. **Innovations:** " + (innovations or "Not specified.") + "\n")
            lines.append("4. **Basic Usage:** " + (basic_usage or "Not specified.") + "\n")
            lines.append("5. **Summary:** " + (summary_text or "Not specified.") + "\n")

            lines.append("---\n\n")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify GitHub repositories into content categories using the selected LLM.")
    parser.add_argument(
        "--from-readme",
        type=str,
        default=None,
        help="Parse repositories from an existing generated README (e.g. README.md) instead of calling GitHub API.",
    )
    parser.add_argument(
        "--from-summaries",
        action="store_true",
        default=False,
        help="Parse repositories from repo_summaries.json. Uses existing AI summaries for classification.",
    )
    parser.add_argument("--max-repos", type=int, default=None, help="Limit number of repos (overrides config/env MAX_REPOS).")
    parser.add_argument("--min-categories", type=int, default=5, help="Min number of categories (default: 5).")
    parser.add_argument("--max-categories", type=int, default=8, help="Max number of categories (default: 8).")
    parser.add_argument(
        "--min-repos-per-category",
        type=int,
        default=None,
        help="Optional: enforce each non-'Other' category has at least X repos by pruning tiny categories into Other (default: read config 'content_min_repos_per_category', 0 disables).",
    )
    parser.add_argument("--sample-size", type=int, default=60, help="How many repos to sample to design taxonomy (default: 60).")
    parser.add_argument("--batch-size", type=int, default=30, help="Repos per LLM classification call (default: 30).")
    parser.add_argument("--out-json", type=str, default="repo_categories.json", help="Output JSON filename.")
    parser.add_argument("--out-md", type=str, default="README.md", help="Output Markdown filename (default: README.md).")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=None,
        help="Override update mode: missing_only or all (also supports env MYGITSTAR_UPDATE_MODE).",
    )

    args = parser.parse_args()

    if args.update_mode is not None:
        global update_mode
        update_mode = normalize_update_mode(args.update_mode)

    # 优先级：命令行参数 > config.yaml > 默认值
    # batch_size
    passed_batch = any(arg.startswith("--batch-size") for arg in sys.argv)
    if not passed_batch and isinstance(config, dict):
        classify_bs = config.get("classify_batch_size")
        if classify_bs is not None:
            try:
                args.batch_size = int(classify_bs)
            except Exception:
                pass
        else:
            batch_bs = config.get("batch_size")
            if batch_bs is not None:
                try:
                    args.batch_size = int(batch_bs)
                except Exception:
                    pass
    # sample_size
    passed_sample = any(arg.startswith("--sample-size") for arg in sys.argv)
    if not passed_sample and isinstance(config, dict):
        sample_size = config.get("classify_sample_size")
        if sample_size is not None:
            try:
                args.sample_size = int(sample_size)
            except Exception:
                pass
    # out-json
    passed_outjson = any(arg.startswith("--out-json") for arg in sys.argv)
    if not passed_outjson and isinstance(config, dict):
        out_json = config.get("classify_out_json")
        if out_json:
            args.out_json = str(out_json)
    # out-md
    passed_outmd = any(arg.startswith("--out-md") for arg in sys.argv)
    if not passed_outmd and isinstance(config, dict):
        out_md = config.get("classify_out_md")
        if out_md:
            args.out_md = str(out_md)

    out_json_path = args.out_json
    if not os.path.isabs(out_json_path):
        out_json_path = os.path.join(REPO_ROOT, out_json_path)

    out_md_path = args.out_md
    if not os.path.isabs(out_md_path):
        out_md_path = os.path.join(REPO_ROOT, out_md_path)

    # If user didn't pass category range flags, allow config.yaml to override defaults.
    # (Workflow always passes explicit values; this mainly improves local runs.)
    passed_min = "--min-categories" in sys.argv
    passed_max = "--max-categories" in sys.argv
    if isinstance(config, dict):
        if not passed_min and config.get("content_min_categories") is not None:
            try:
                v = config.get("content_min_categories")
                if v is not None:
                    args.min_categories = int(v)
            except Exception:
                pass
        if not passed_max and config.get("content_max_categories") is not None:
            try:
                v = config.get("content_max_categories")
                if v is not None:
                    args.max_categories = int(v)
            except Exception:
                pass

    if args.min_categories < 1:
        print("--min-categories must be >= 1")
        return 2
    if args.max_categories < args.min_categories:
        print("--max-categories must be >= --min-categories")
        return 2

    # NOTE:
    # - For --from-readme mode, we want to classify ALL repos already present in README by default.
    #   So we only apply a limit if the user explicitly passes --max-repos.
    # - For GitHub API mode, we keep the old behavior: env MAX_REPOS / config.max_repos can limit the run.
    max_repos = args.max_repos
    if max_repos is None and not args.from_readme:
        env_mr = os.environ.get("MAX_REPOS")
        if env_mr:
            try:
                mr = int(env_mr)
                if mr > 0:
                    max_repos = mr
            except Exception:
                max_repos = None
        if max_repos is None:
            cfg_mr = config.get("max_repos") if isinstance(config, dict) else None
            try:
                if cfg_mr is not None and int(cfg_mr) > 0:
                    max_repos = int(cfg_mr)
            except Exception:
                pass

    if args.from_readme:
        try:
            repos = parse_repos_from_readme(args.from_readme, max_repos=max_repos)
        except Exception as e:
            print(f"Failed to parse repos from README: {e}")
            return 2
    elif args.from_summaries:
        try:
            repos = parse_repos_from_summaries(None, max_repos=max_repos)
        except Exception as e:
            print(f"Failed to parse repos from summaries: {e}")
            return 2
    else:
        try:
            repos = get_starred_repos(max_repos=max_repos)
        except Exception as e:
            print(f"Failed to fetch starred repos: {e}")
            print("\nChecklist:")
            print("- Ensure you have set STARRED_GITHUB_TOKEN in environment variables, or in GitHub Actions Secrets.")
            print("- If config.yaml sets github_username: 0, ensure GITHUB_ACTOR is available (Actions) or set github_username to your username locally.")
            return 2

    if not repos:
        print("No repos fetched.")
        return 2

    # Auto-enrich repos with existing summary data if available
    repos = _enrich_repos_from_summaries(repos)

    # Ensure a consistent content text field exists for both API-fetched repos and README-parsed repos.
    for r in repos:
        try:
            if not str(r.get("content_text") or "").strip():
                r["content_text"] = _build_repo_content_text(r)
        except Exception:
            pass

    min_repos_per_category = _resolve_min_repos_per_category(args.min_repos_per_category)
    effective_max_categories = _compute_effective_max_categories(
        total_repos=len(repos),
        min_repos_per_category=min_repos_per_category,
        min_categories=args.min_categories,
        max_categories=args.max_categories,
    )
    if effective_max_categories != args.max_categories:
        print(
            f"[info] Adjusted max_categories {args.max_categories} -> {effective_max_categories} due to content_min_repos_per_category={min_repos_per_category} (total_repos={len(repos)})"
        )
        args.max_categories = effective_max_categories

    taxonomy: Optional[Taxonomy] = None
    assignment_map: Dict[Any, str] = {}
    all_ids = {r.get("id") for r in repos}
    repos_to_classify: List[Dict[str, Any]] = []
    reused_count = 0

    # Step A: try to REUSE the existing taxonomy + per-repo assignments
    # from repo_categories.json. This is the main API-saver for the
    # classification step.
    cache_max_age_days = 0
    new_repo_ratio_threshold = 0.3
    stability_threshold = 0.7
    try:
        if isinstance(config, dict):
            cache_max_age_days = int(config.get("taxonomy_cache_max_age_days", 0) or 0)
            new_repo_ratio_threshold = float(config.get("taxonomy_cache_new_repo_ratio", 0.3) or 0.3)
            stability_threshold = float(config.get("taxonomy_cache_stability_threshold", 0.7) or 0.7)
    except Exception:
        pass

    if cache_max_age_days > 0 and os.path.exists(out_json_path):
        cached_tax, cached_assigns, cached_summaries, cached_at = load_existing_categories(out_json_path)
        should_refresh, refresh_reason = _should_refresh_taxonomy(
            cached_tax, cached_assigns, cached_at, repos,
            cache_max_age_days, args.min_categories, args.max_categories,
            min_repos_per_category, new_repo_ratio_threshold,
        )
        if cached_tax and not should_refresh:
            taxonomy = cached_tax
            # Reuse per-repo assignments where still valid.
            # Track repos previously in "Other"/"Unknown" for priority re-classification.
            priority_repos: List[Dict[str, Any]] = []
            for r in repos:
                rid = r.get("id")
                full_name = str(r.get("full_name") or "").strip()
                cached_cid = cached_assigns.get(full_name)
                if cached_cid and _is_assignment_still_valid(cached_cid, r, taxonomy):
                    assignment_map[rid] = cached_cid
                    reused_count += 1
                else:
                    # Check if this repo was previously in "Other"/"Unknown" and now has updated summary
                    if cached_cid and cached_cid.lower() in ("other", "unknown"):
                        current_summary = (r.get("summary") or "").strip()
                        brief = (r.get("brief_intro") or "").strip()
                        cached_summary = cached_summaries.get(full_name, "")
                        # Re-classify if summary was updated (different from cached)
                        if (current_summary or brief) and current_summary != cached_summary:
                            priority_repos.append(r)
                            continue
                    repos_to_classify.append(r)
            # Prioritize "Other"/"Unknown" repos with updated summaries - process them first
            # so they get classified in earlier batches
            repos_to_classify = priority_repos + repos_to_classify
            if priority_repos:
                print(f"[PRIORITY] {len(priority_repos)} repos previously in 'Other'/'Unknown' have updated summaries, will be classified first")
            print(f"[REUSE] Taxonomy cache valid; reused {reused_count}/{len(repos)} assignments; need LLM for {len(repos_to_classify)} repos")
        else:
            reason = refresh_reason if should_refresh else "no cached taxonomy"
            print(f"[REFRESH] Taxonomy cache invalid ({reason}); will redesign taxonomy")
            repos_to_classify = repos
    else:
        repos_to_classify = repos

    # `unknown_repos` is needed regardless of which path we take (cache
    # reuse vs. fresh taxonomy), so compute it once here.
    unknown_repos = [r for r in repos if r.get("is_unknown")]

    if taxonomy is None:
        known_repos = [r for r in repos if not r.get("is_unknown")]
        sample_size = max(5, min(args.sample_size, len(known_repos)))
        sample = _sample_repos_for_taxonomy(known_repos, sample_size)
        taxonomy_prompt = build_taxonomy_prompt(sample, min_categories=args.min_categories, max_categories=args.max_categories)
        print(
            f"Designing taxonomy from {len(sample)} sample repos (min_categories={args.min_categories}, max_categories={args.max_categories})..."
        )
        try:
            raw_tax = call_llm_json(taxonomy_prompt, attempts=2)
        except RateLimitAbort as e:
            print(f"[FATAL] {e}")
            return 4
        except Exception as e:
            print(f"Failed to call LLM for taxonomy: {e}")
            print("\nChecklist:")
            print("- model_choice=copilot requires STARRED_GITHUB_TOKEN")
            print("- model_choice=openrouter requires OPENROUTER_API_KEY")
            print("- model_choice=gemini requires GEMINI_API_KEY")
            return 3

        try:
            taxonomy = _normalize_taxonomy(raw_tax, min_categories=args.min_categories, max_categories=args.max_categories)
            # If the model ignored constraints and produced mostly language buckets (filtered out), retry once with stronger warning.
            if len(taxonomy.categories) < args.min_categories:
                retry_prompt = taxonomy_prompt + "\n\nIMPORTANT: Your previous output likely used programming-language categories. Redesign taxonomy strictly by CONTENT domains."
                raw_tax2 = call_llm_json(retry_prompt, attempts=1)
                taxonomy = _normalize_taxonomy(raw_tax2, min_categories=args.min_categories, max_categories=args.max_categories)
        except Exception as e:
            print(f"Failed to normalize taxonomy JSON: {e}")
            return 3

        print("Taxonomy:")
        for c in taxonomy.categories:
            print(f"  {c['id']}: {c['name']}")

    unknown_id = "Unknown"
    unknown_cat_exists = any(c.get("id", "").lower() == "unknown" or c.get("name", "").lower() == "unknown" for c in taxonomy.categories)
    if not unknown_cat_exists:
        taxonomy.categories.append({
            "id": unknown_id,
            "name": "Unknown",
            "description": "Repositories with insufficient summary data (empty Brief Introduction, Innovations, and Summary)."
        })
        print(f"Added 'Unknown' category for {len(unknown_repos)} repos with missing summaries.")

    for r in unknown_repos:
        rid = r.get("id")
        if rid:
            assignment_map[rid] = unknown_id

    # Classify repos in batches
    if repos_to_classify:
        print(f"Classifying {len(repos_to_classify)} repos in batches of {args.batch_size}...")
        for i, batch in enumerate(chunk_list(repos_to_classify, args.batch_size), start=1):
            prompt = build_classification_prompt(taxonomy, batch)
            try:
                raw = call_llm_json(prompt, attempts=2)
                assignments = _parse_assignments(raw)
            except RateLimitAbort as e:
                print(f"[FATAL] {e}")
                return 4
            except Exception as e:
                print(f"[Batch {i}] failed to parse assignments: {e}")
                assignments = []

            # validate and fill
            valid_category_ids = {c["id"] for c in taxonomy.categories}
            name_to_id = {
                str(c.get("name", "")).strip().lower(): str(c.get("id"))
                for c in taxonomy.categories
                if str(c.get("name", "")).strip()
            }
            other_id = next(
                (c["id"] for c in taxonomy.categories if str(c["name"]).strip().lower() == "other"),
                taxonomy.categories[-1]["id"],
            )
            for a in assignments:
                rid = a.get("id")
                cid = a.get("category_id") or a.get("category") or a.get("categoryId")
                if not rid or not cid:
                    continue
                cid_str = str(cid).strip()
                if cid_str not in valid_category_ids:
                    cid_name = cid_str.strip().lower()
                    mapped = name_to_id.get(cid_name)
                    if mapped:
                        cid_str = mapped
                    elif cid_str.isdigit() and f"C{cid_str}" in valid_category_ids:
                        cid_str = f"C{cid_str}"
                    else:
                        cid_str = other_id
                assignment_map[rid] = cid_str

            # Fill missing in this batch as Other
            batch_ids = {r.get("id") for r in batch}
            for rid in batch_ids:
                if rid not in assignment_map:
                    assignment_map[rid] = other_id

            print(f"  batch {i}/{(len(repos_to_classify) + args.batch_size - 1) // args.batch_size} done")
            time.sleep(RATE_LIMIT_DELAY)

    # Ensure every repo is assigned
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])
    for rid in all_ids:
        if rid not in assignment_map:
            assignment_map[rid] = other_id

    # Optional: prune tiny categories into Other to keep buckets meaningful.
    taxonomy, assignment_map = _apply_min_repos_per_category(
        taxonomy,
        repos,
        assignment_map,
        min_repos_per_category=min_repos_per_category,
        min_categories=args.min_categories,
    )
    other_id = next((c["id"] for c in taxonomy.categories if c["name"].lower() == "other"), taxonomy.categories[-1]["id"])

    # Check taxonomy quality and output warnings if issues found
    quality_issues = _check_taxonomy_quality(
        taxonomy,
        assignment_map,
        min_repos_per_category,
        args.min_categories,
        args.max_categories,
    )
    if quality_issues:
        print(f"\n[WARNING] Taxonomy quality issues detected ({len(quality_issues)}):")
        for issue in quality_issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(quality_issues) > 5:
            print(f"  ... and {len(quality_issues) - 5} more issues")
        print("Consider adjusting content_min_categories, content_max_categories, or content_min_repos_per_category in config.yaml\n")

    # Check classification stability
    is_stable, reuse_ratio = _check_classification_stability(
        reused_count, len(repos), len(unknown_repos), stability_threshold
    )
    if not is_stable and reused_count > 0:
        print(f"\n[WARNING] Classification stability is low: {reuse_ratio:.1%} (threshold: {stability_threshold:.1%})")
        print("This may indicate taxonomy drift. Consider redesigning taxonomy by setting taxonomy_cache_max_age_days: 0\n")

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_choice": MODEL_CHOICE,
        "min_categories": args.min_categories,
        "max_categories": args.max_categories,
        "min_repos_per_category": min_repos_per_category,
        "taxonomy": {"categories": taxonomy.categories},
        "assignments": [
            {
                "id": r.get("id"),
                "full_name": r.get("full_name"),
                "category_id": assignment_map.get(r.get("id"), other_id),
                # 'reused' if we kept the cached assignment, 'ai' if we just
                # asked the LLM, 'unknown' if the repo had no summary.
                "classification_source": (
                    "reused" if r.get("id") in assignment_map and r.get("id") not in {x.get("id") for x in repos_to_classify}
                    else ("unknown" if r in unknown_repos else "ai")
                ),
                # Store summary for Other/Unknown repos to detect updates next run
                "summary": (
                    (r.get("summary") or "").strip()
                    if assignment_map.get(r.get("id"), other_id).lower() in ("other", "unknown")
                    else None
                ),
            }
            for r in repos
        ],
        "stats": {
            "total_repos": len(repos),
            "reused": reused_count,
            "llm_classified": len(repos) - reused_count - len(unknown_repos),
            "unknown": len(unknown_repos),
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    generated_at_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    username = get_current_account_optional()
    lang = str(config.get("language", "en")).strip().lower()
    md = render_markdown(taxonomy, repos, assignment_map, config, MODEL_CHOICE, username, generated_at_str, lang)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())