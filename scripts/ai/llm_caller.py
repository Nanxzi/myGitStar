import json
import os
import random
import re
import time
from typing import Any, Dict, Optional

import requests

from scripts.core.throttle import SimpleThrottle


API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


class RateLimitAbort(RuntimeError):
    """Raised when the LLM backend returns too many 429s or the per-run
    API-call budget is exhausted. Carries an optional `results` payload so
    the caller can still persist already-completed summaries before exiting.
    """

    def __init__(self, message: str, results: Optional[Any] = None):
        super().__init__(message)
        self.results = results


MAX_CONSECUTIVE_429 = 5
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


def make_api_request(
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    retries: int,
    retry_delay: float,
    throttle: Optional[SimpleThrottle] = None,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    print(f"[DEBUG] make_api_request: ENTERING, url={url}, retries={retries}, timeout={timeout}", flush=True)
    for attempt in range(retries):
        try:
            if throttle:
                try:
                    throttle.wait()
                except Exception:
                    pass
            print(f"[DEBUG] make_api_request: attempt {attempt + 1}/{retries}, calling requests.post...", flush=True)
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            print(f"[DEBUG] make_api_request: requests.post returned, status_code={resp.status_code}", flush=True)

            if resp.status_code == 429:
                retry_after = None
                try:
                    ra = resp.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = int(ra)
                except Exception:
                    retry_after = None

                if retry_after is not None and retry_after > 600:
                    return {
                        "error": {"code": 429, "message": f"Too Many Requests, Retry-After={retry_after}s, skipped batch."},
                        "status_code": 429,
                    }

                if attempt < retries - 1:
                    wait = retry_after if retry_after else (retry_delay * (2**attempt) + random.uniform(0, 1))
                    print(f"[429] Rate limited, sleep {wait:.1f}s then retry ({attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue
                return {"error": {"code": 429, "message": "Too Many Requests"}, "status_code": 429}

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
                wait = retry_delay * (2**attempt)
                print(f"[HTTPError] {e}; sleep {wait:.1f}s then retry")
                time.sleep(wait)
                continue
            print(f"[HTTPError] final: {e}")
            return {"error": {"code": "http_error", "message": str(e)}, "status_code": "http_error"}
        except Exception as e:
            if attempt < retries - 1:
                wait = retry_delay * (2**attempt) + random.uniform(0, 1)
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
    if text is None:
        raise ValueError("empty response")
    s = str(text).strip()

    s = re.sub(r"^```(json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)

    try:
        return json.loads(s)
    except Exception:
        pass

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


def call_llm(
    prompt: str,
    config: Dict[str, Any],
    github_token: str,
    openrouter_api_key: str,
    gemini_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    api_request_func: callable,
    fallback_on_429: bool = False,
) -> str:
    model_choice = config.get("model_choice", "copilot")
    if not model_choice:
        model_choice = os.environ.get("MYGITSTAR_MODEL_CHOICE", "copilot")

    if model_choice == "copilot":
        if not github_token:
            raise RuntimeError("Missing STARRED_GITHUB_TOKEN (required for copilot backend).")
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/json",
            "X-GitHub-Api-Version": "2023-07-01",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("GITHUB_COPILOT_MODEL", default_copilot_model) or "openai/gpt-4o-mini"
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1800,
            "temperature": 0.2,
        }
        resp = api_request_func(API_ENDPOINTS["copilot"], headers, data, retries=3, retry_delay=2.0)

        if (
            fallback_on_429
            and isinstance(resp, dict)
            and isinstance(resp.get("error"), dict)
            and resp.get("error", {}).get("code") == 429
        ):
            print("[WARN] Copilot rate-limited (429). Trying fallback backend...")
            if openrouter_api_key:
                model_name2 = default_openrouter_model or "openai/gpt-4o-mini"
                headers2 = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}
                data2 = {"model": model_name2, "messages": [{"role": "user", "content": prompt}]}
                resp2 = api_request_func(API_ENDPOINTS["openrouter"], headers2, data2, retries=3, retry_delay=2.0)
                _raise_if_error_response(resp2, backend="OpenRouter")
                text2 = _get_llm_text(resp2)
                if not str(text2).strip():
                    raise RuntimeError("OpenRouter returned empty content")
                return text2

            if gemini_api_key:
                model_name2 = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-2.0-flash"
                model_path2 = str(model_name2).strip()
                if model_path2.startswith("models/"):
                    model_path2 = model_path2[len("models/"):]

                api_url2 = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path2}:generateContent"
                request_url2 = f"{api_url2}?key={gemini_api_key}"
                headers2 = {
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "GitHub Star Content Classifier/1.0",
                    "X-Goog-Api-Key": gemini_api_key,
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
                resp2 = api_request_func(
                    request_url2,
                    headers2,
                    payload2,
                    retries=3,
                    retry_delay=2.0,
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
        if not openrouter_api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY (required for openrouter backend).")
        model_name = default_openrouter_model or "openai/gpt-4o-mini"
        print(f"[DEBUG] OpenRouter model: {model_name}, API key prefix: {openrouter_api_key[:5]}...")
        headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5120,
            "temperature": 0.2,
        }
        resp = api_request_func(API_ENDPOINTS["openrouter"], headers, data, retries=3, retry_delay=2.0)
        _raise_if_error_response(resp, backend="OpenRouter")
        text = _get_llm_text(resp)
        if not str(text).strip():
            raise RuntimeError("OpenRouter returned empty content")
        return text

    if model_choice == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (required for gemini backend).")

        model_name = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-2.0-flash"
        model_path = str(model_name).strip()
        if model_path.startswith("models/"):
            model_path = model_path[len("models/"):]

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        request_url = f"{api_url}?key={gemini_api_key}"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "GitHub Star Content Classifier/1.0",
            "X-Goog-Api-Key": gemini_api_key,
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

        resp = api_request_func(
            request_url,
            headers,
            payload,
            retries=3,
            retry_delay=2.0,
        )
        _raise_if_error_response(resp, backend="Gemini")

        if not isinstance(resp, dict):
            raise RuntimeError("Gemini request failed: invalid response type")

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

    raise RuntimeError(f"Unsupported model_choice: {model_choice}")


def call_llm_json(
    prompt: str,
    config: Dict[str, Any],
    github_token: str,
    openrouter_api_key: str,
    gemini_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    api_request_func: callable,
    fallback_on_429: bool = False,
    attempts: int = 2,
) -> Any:
    last_error: Optional[Exception] = None
    for i in range(1, max(1, attempts) + 1):
        p = prompt
        if i > 1:
            p = (
                prompt
                + "\n\nIMPORTANT: Return STRICT JSON only. Do not include markdown fences, explanations, or extra text."
            )
        try:
            text = call_llm(
                p,
                config,
                github_token,
                openrouter_api_key,
                gemini_api_key,
                default_copilot_model,
                default_openrouter_model,
                default_gemini_model,
                api_request_func,
                fallback_on_429,
            )
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
