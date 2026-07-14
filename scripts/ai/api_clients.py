import os
import time
from typing import Any, Dict, Optional

from scripts.ai.llm_caller import make_api_request, RateLimitAbort


API_ENDPOINTS = {
    "copilot": "https://models.github.ai/inference/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    "modelscope": "https://api-inference.modelscope.cn/v1/chat/completions",
}


def _extract_prompt(repo: Dict) -> str:
    if isinstance(repo, dict):
        if "prompt" in repo:
            return repo["prompt"]
    return repo.get("prompt", "") if isinstance(repo, dict) else ""


def copilot_summarize(
    repo: Dict[str, Any],
    github_token: str,
    default_copilot_model: str,
    api_request_func: callable,
) -> Optional[str]:
    if not github_token:
        return None
    try:
        prompt = _extract_prompt(repo)
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
            "max_tokens": 600,
            "temperature": 0.4,
        }
        response = api_request_func(API_ENDPOINTS["copilot"], headers, data)
        if response and isinstance(response, dict) and response.get("error"):
            err = response["error"]
            if err.get("code") == "RateLimitReached":
                return f"Copilot API限额已用尽：{err.get('message', 'RateLimitReached')}"
        content = None
        if response:
            choices = response.get("choices", [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if message and isinstance(message, dict):
                    content = message.get("content", "")
                elif "content" in choices[0]:
                    content = choices[0]["content"]
            if content is not None:
                content = str(content).strip()
        return content if content else None
    except RateLimitAbort:
        # Let rate limit abort propagate to caller
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] copilot_summarize: Exception: {type(e).__name__}: {e}", flush=True)
        print(f"[ERROR] copilot_summarize: Exception traceback: {traceback.format_exc()}", flush=True)
        return None


def openrouter_summarize(
    repo: Dict[str, Any],
    openrouter_api_key: str,
    default_openrouter_model: str,
    api_request_func: callable,
) -> Optional[str]:
    import sys
    print("[DEBUG] openrouter_summarize: ENTERING function", flush=True)
    sys.stdout.flush()
    if not openrouter_api_key:
        print("[DEBUG] openrouter_summarize: no API key", flush=True)
        return None
    try:
        prompt = _extract_prompt(repo)
        print(f"[DEBUG] openrouter_summarize: model={default_openrouter_model}, prompt_length={len(prompt)}", flush=True)
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": default_openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5120,
            "temperature": 0.2,
        }
        print(f"[DEBUG] openrouter_summarize: calling api_request_func...", flush=True)
        response = api_request_func(API_ENDPOINTS["openrouter"], headers, data)
        print(f"[DEBUG] openrouter_summarize: api_request_func returned, response={type(response)}", flush=True)
        content = None
        if response:
            print(f"[DEBUG] openrouter_summarize: response keys={list(response.keys()) if isinstance(response, dict) else 'N/A'}", flush=True)
            choices = response.get("choices", [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if message and isinstance(message, dict):
                    content = message.get("content", "")
                elif "content" in choices[0]:
                    content = choices[0]["content"]
            if content is not None:
                content = str(content).strip()
        return content if content else None
    except Exception as e:
        import traceback
        print(f"[ERROR] openrouter_summarize: Exception: {type(e).__name__}: {e}", flush=True)
        print(f"[ERROR] openrouter_summarize: Exception traceback: {traceback.format_exc()}", flush=True)
        return None


def gemini_summarize(
    repo: Dict[str, Any],
    gemini_api_key: str,
    default_gemini_model: str,
    config: Dict[str, Any],
    api_request_func: callable,
) -> Optional[str]:
    if not gemini_api_key:
        return None

    prompt = _extract_prompt(repo)
    if not prompt.strip():
        return None

    model_name = os.environ.get("GEMINI_MODEL", default_gemini_model) or "gemini-pro"
    model_path = str(model_name).strip()
    if model_path.startswith("models/"):
        model_path = model_path[len("models/"):]
    model_path = model_path.strip()

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
    request_url = f"{api_url}?key={gemini_api_key}"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "GitHub Star Summary Bot/1.0",
        "X-Goog-Api-Key": gemini_api_key,
    }

    temperature = config.get("gemini_temperature", 0.4)
    max_output_tokens = config.get("gemini_max_output_tokens", 800)
    payload = {
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.8,
            "topK": 40,
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }

    gen_retries = int(config.get("gemini_generation_retries", 3))
    gen_backoff = int(config.get("gemini_retry_backoff", 5))
    base_max_tokens = int(config.get("gemini_max_output_tokens", max_output_tokens))

    final_content = None
    for attempt in range(1, gen_retries + 1):
        attempt_max_tokens = min(base_max_tokens + (attempt - 1) * 200, 2048)
        payload["generationConfig"]["maxOutputTokens"] = attempt_max_tokens

        response = api_request_func(request_url, headers, payload)

        if not response or not isinstance(response, dict):
            if attempt < gen_retries:
                wait = gen_backoff * attempt
                time.sleep(wait)
                continue
            else:
                return None

        if "error" in response:
            error = response["error"]
            error_code = error.get("code")
            error_msg = error.get("message", "未知错误")
            if error_code in (429, 503):
                if attempt < gen_retries:
                    wait = gen_backoff * attempt
                    time.sleep(wait)
                    continue
            if error_code == 401:
                return "Gemini API Key 无效或无权限"
            elif error_code == 404:
                return f"Gemini 模型 {model_path} 不存在"
            elif error_code == 400:
                return "Gemini 请求参数格式错误"
            else:
                return f"Gemini API 错误: {error_msg}"

        content = ""
        truncated = False
        try:
            candidates = response.get("candidates", [])
            for candidate in candidates:
                finish = candidate.get("finishReason")
                content_obj = candidate.get("content", {})
                parts = content_obj.get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        content += part["text"].strip() + "\n"
                if finish == "MAX_TOKENS":
                    truncated = True
                break

            if not content:
                choices = response.get("choices", [])
                for choice in choices:
                    if isinstance(choice, dict):
                        content = choice.get("message", {}).get("content", "") or choice.get("text", "")
                        if content:
                            break

            content = content.strip()
        except Exception:
            content = ""

        if content and not truncated:
            final_content = content
            break

        if attempt < gen_retries:
            wait = gen_backoff * attempt
            time.sleep(wait)
            continue
        else:
            if content:
                final_content = content
            else:
                return None

    return final_content


def modelscope_summarize(
    repo: Dict[str, Any],
    modelscope_api_key: str,
    default_modelscope_model: str,
    api_request_func: callable,
) -> Optional[str]:
    import sys
    print("[DEBUG] modelscope_summarize: ENTERING function", flush=True)
    sys.stdout.flush()
    if not modelscope_api_key:
        print("[DEBUG] modelscope_summarize: no API key", flush=True)
        return None
    try:
        prompt = _extract_prompt(repo)
        print(f"[DEBUG] modelscope_summarize: model={default_modelscope_model}, prompt_length={len(prompt)}", flush=True)
        headers = {
            "Authorization": f"Bearer {modelscope_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": default_modelscope_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5120,
            "temperature": 0.2,
        }
        print(f"[DEBUG] modelscope_summarize: calling api_request_func...", flush=True)
        response = api_request_func(API_ENDPOINTS["modelscope"], headers, data)
        print(f"[DEBUG] modelscope_summarize: api_request_func returned, response={type(response)}", flush=True)
        content = None
        if response:
            print(f"[DEBUG] modelscope_summarize: response keys={list(response.keys()) if isinstance(response, dict) else 'N/A'}", flush=True)
            choices = response.get("choices", [{}])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if message and isinstance(message, dict):
                    content = message.get("content", "")
                elif "content" in choices[0]:
                    content = choices[0]["content"]
            if content is not None:
                content = str(content).strip()
        return content if content else None
    except Exception as e:
        import traceback
        print(f"[ERROR] modelscope_summarize: Exception: {type(e).__name__}: {e}", flush=True)
        print(f"[ERROR] modelscope_summarize: Exception traceback: {traceback.format_exc()}", flush=True)
        return None


def create_summarize_func(
    model_choice: str,
    github_token: str,
    openrouter_api_key: str,
    gemini_api_key: str,
    modelscope_api_key: str,
    default_copilot_model: str,
    default_openrouter_model: str,
    default_gemini_model: str,
    default_modelscope_model: str,
    language: str,
    config: dict,
    throttle: Any,
    request_timeout: float,
    request_retry_delay: float,
    retry_attempts: int,
    api_call_counter: callable,
):
    def make_request(url, headers, data):
        return make_api_request(url, headers, data, retry_attempts, request_retry_delay, throttle, request_timeout)

    if model_choice == "copilot":
        def summarize(repo: Dict) -> Optional[str]:
            result = copilot_summarize(
                repo,
                github_token,
                default_copilot_model,
                make_request,
            )
            api_call_counter()
            return result
    elif model_choice == "openrouter":
        print(f"[DEBUG] create_summarize_func: openrouter mode, api_key={'set' if openrouter_api_key else 'EMPTY'}, model={default_openrouter_model}", flush=True)
        def summarize(repo: Dict) -> Optional[str]:
            result = openrouter_summarize(
                repo,
                openrouter_api_key,
                default_openrouter_model,
                make_request,
            )
            api_call_counter()
            return result
    elif model_choice == "gemini":
        def summarize(repo: Dict) -> Optional[str]:
            result = gemini_summarize(
                repo,
                gemini_api_key,
                default_gemini_model,
                config,
                make_request,
            )
            api_call_counter()
            return result
    elif model_choice == "modelscope":
        print(f"[DEBUG] create_summarize_func: modelscope mode, api_key={'set' if modelscope_api_key else 'EMPTY'}, model={default_modelscope_model}", flush=True)
        def summarize(repo: Dict) -> Optional[str]:
            result = modelscope_summarize(
                repo,
                modelscope_api_key,
                default_modelscope_model,
                make_request,
            )
            api_call_counter()
            return result
    elif model_choice == "lmstudio":
        from scripts.ai.lmstudio_client import create_lmstudio_summarize_func
        lmstudio_model = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")
        return create_lmstudio_summarize_func(model=lmstudio_model)
    elif model_choice == "ollama":
        from scripts.ai.ollama_client import create_ollama_summarize_func
        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
        return create_ollama_summarize_func(model=ollama_model)
    else:
        raise ValueError(f"不支持的模型选择: {model_choice}")

    return summarize
