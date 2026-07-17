import concurrent.futures
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# Patterns that indicate prompt template leakage from the LLM
_PROMPT_LEAK_PATTERNS = [
    r"\*\*\s*\(Based on README.*?\)\s*\n?",
    r"\*\*\s*\(within \d+ words?\)\s*\n?",
    r"\*\*\s*\(50字以内\)\s*\n?",
    r"\*\*\s*\(no description available\)\s*\n?",
    r"\*\*\s*\(omit if none\)\s*\n?",
    r"\*\*\s*\(如无则略\)\s*\n?",
    r"\*\*\s*\(如无则写 Not specified\)\s*\n?",
    r"\*\*\s*\(One sentence.*?\)\s*\n?",
    r"\*\*\s*\(一句话总结.*?\)\s*\n?",
    r"\*\*\s*\(Briefly describe.*?\)\s*\n?",
    r"\*\*\s*\(简述.*?\)\s*\n?",
    r"\*\*\s*\(Provide the simplest.*?\)\s*\n?",
    r"\*\*\s*\(给出最简.*?\)\s*\n?",
]


def _clean_prompt_leak(text: str) -> str:
    """Remove prompt template artifacts leaked by the LLM into field values."""
    if not text:
        return ""
    for pattern in _PROMPT_LEAK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Remove leading ** markers that LLM sometimes adds
    text = re.sub(r"^\*\*\s*", "", text)
    text = re.sub(r"\n\*\*\s*", "\n", text)
    # Remove leading markdown heading markers (# ## ### etc.)
    text = re.sub(r"^#+\s*", "", text)
    return text.strip()


def _repo_key(repo: Dict) -> str:
    return str(repo.get("full_name") or repo.get("Repository Name") or "").strip()


def generate_summarize_prompt(repo: Dict[str, Any], language: str = "zh", readme_content: str = "") -> str:
    repo_name = repo["full_name"]
    desc = repo.get("description") or ""
    url = repo.get("html_url") or ""

    # Brief Introduction = original description (no LLM needed)
    brief_intro = desc if desc else "(no description available)"

    if language == "zh":
        readme_section = ""
        if readme_content:
            readme_section = f"\n**仓库 README 内容（节选）：**\n{readme_content}\n"

        return (
            f"请对以下 GitHub 仓库进行内容总结，按如下格式输出：\n"
            f"1. **仓库名称：** {repo_name}\n"
            f"2. **简要介绍：** {brief_intro}\n"
            f"3. **创新点：** （基于 README 内容，简述本仓库最有特色的地方，50字以内）\n"
            f"4. **简单用法：** （基于 README 内容，给出最简关键用法或调用示例，如无则写 Not specified）\n"
            f"5. **总结：** （一句话总结它的用途/价值，50字以内）\n"
            f"**仓库描述：** {desc}\n"
            f"**仓库地址：** {url}\n"
            f"{readme_section}"
        )
    else:
        readme_section = ""
        if readme_content:
            readme_section = f"\n**Repository README content (excerpt):**\n{readme_content}\n"

        return (
            f"Please summarize the following GitHub repository in the specified format:\n"
            f"1. **Repository Name:** {repo_name}\n"
            f"2. **Brief Introduction:** {brief_intro}\n"
            f"3. **Innovations:** (Based on README, briefly describe the most distinctive features, within 50 words)\n"
            f"4. **Basic Usage:** (Based on README, provide the simplest key usage or example, write 'Not specified' if none)\n"
            f"5. **Summary:** (One sentence summarizing its purpose/value, within 50 words)\n"
            f"**Repository Description:** {desc}\n"
            f"**Repository URL:** {url}\n"
            f"{readme_section}"
        )


def generate_combined_summarize_prompt(repos: List[Dict[str, Any]], language: str = "zh") -> str:
    if language == "zh":
        repo_list = []
        for i, repo in enumerate(repos, 1):
            repo_name = repo["full_name"]
            desc = repo.get("description") or ""
            url = repo.get("html_url") or ""
            readme = repo.get("readme_content") or ""
            entry = (
                f"## 仓库 {i}\n"
                f"- 仓库名称: {repo_name}\n"
                f"- 描述: {desc}\n"
                f"- 地址: {url}"
            )
            if readme:
                entry += f"\n- README 内容（节选）:\n{readme}"
            repo_list.append(entry)

        return (
            f"你是一个GitHub仓库总结助手。请对以下 {len(repos)} 个仓库分别进行总结，\n"
            f"每个仓库按以下固定格式输出（注意：仓库名称必须与输入完全一致）：\n\n"
            "输出格式（必须是JSON数组）：\n"
            "```json\n"
            "[\n"
            '  {"repo": "owner/repo", "Repository Name": "...", "Repository URL": "...", "Brief Introduction": "...", "Innovations": "...", "Basic Usage": "...", "Summary": "..."},\n'
            "]\n"
            "```\n\n"
            "要求：\n"
            "- Repository Name: 仓库全名（必须与输入完全一致）\n"
            "- Repository URL: 仓库地址\n"
            "- Brief Introduction: 直接使用仓库的原始描述（不要改写）\n"
            "- Innovations: 基于 README 内容，创新点（50字以内）\n"
            "- Basic Usage: 基于 README 内容，简单用法（如无则写 Not specified）\n"
            "- Summary: 一句话总结（50字以内）\n"
            "- 只输出JSON数组，不要输出其他内容\n\n"
            "## 待总结的仓库：\n"
            + "\n\n".join(repo_list)
        )
    else:
        repo_list = []
        for i, repo in enumerate(repos, 1):
            repo_name = repo["full_name"]
            desc = repo.get("description") or ""
            url = repo.get("html_url") or ""
            readme = repo.get("readme_content") or ""
            entry = (
                f"## Repository {i}\n"
                f"- Name: {repo_name}\n"
                f"- Description: {desc}\n"
                f"- URL: {url}"
            )
            if readme:
                entry += f"\n- README content (excerpt):\n{readme}"
            repo_list.append(entry)

        return (
            f"You are a GitHub repository summarization assistant. Please summarize the following {len(repos)} repositories.\n"
            f"Each repository must follow this exact format (note: repository name must match exactly):\n\n"
            "Output format (must be JSON array):\n"
            "```json\n"
            "[\n"
            '  {"repo": "owner/repo", "Repository Name": "...", "Repository URL": "...", "Brief Introduction": "...", "Innovations": "...", "Basic Usage": "...", "Summary": "..."},\n'
            "]\n"
            "```\n\n"
            "Requirements:\n"
            "- Repository Name: full repository name (must match exactly)\n"
            "- Repository URL: repository URL\n"
            "- Brief Introduction: use the original description verbatim (do NOT rewrite)\n"
            "- Innovations: based on README, key innovations (within 50 words)\n"
            "- Basic Usage: based on README, basic usage (write 'Not specified' if none)\n"
            "- Summary: one sentence summary (within 50 words)\n"
            "- Output only JSON array, nothing else\n\n"
            "## Repositories to summarize:\n"
            + "\n\n".join(repo_list)
        )


def parse_combined_summaries(response_text: str, repos: List[Dict[str, Any]]) -> Dict[str, Dict]:
    import json

    results: Dict[str, Dict] = {}
    for repo in repos:
        results[repo["full_name"]] = {}

    if not response_text:
        return results

    text = response_text.strip()

    json_match = None
    for pattern in [r"```json\s*(\[[\s\S]*?)\s*```", r"(\[[\s\S]*?\])"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    json_match = data
                    break
            except json.JSONDecodeError:
                continue

    if not json_match:
        try:
            arr_start = text.find("[")
            arr_end = text.rfind("]")
            if arr_start != -1 and arr_end != -1:
                json_str = text[arr_start:arr_end+1]
                json_match = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            print(f"[WARN] JSON parse failed for batch, repos: {[r['full_name'] for r in repos]}")
            pass

    if json_match and isinstance(json_match, list):
        for item in json_match:
            if isinstance(item, dict):
                repo_name = item.get("repo", "") or item.get("Repository Name", "")
                if not repo_name:
                    continue

                brief_intro = (item.get("Brief Introduction") or item.get("简要介绍") or "")
                innovations = (item.get("Innovations") or item.get("创新点") or "")
                basic_usage = (item.get("Basic Usage") or item.get("简单用法") or "")
                summary = (item.get("Summary") or item.get("总结") or "")

                # Clean prompt leak artifacts from all fields
                brief_intro = _clean_prompt_leak(brief_intro)
                innovations = _clean_prompt_leak(innovations)
                basic_usage = _clean_prompt_leak(basic_usage)
                summary = _clean_prompt_leak(summary)

                full_entry = {
                    "Repository Name": repo_name,
                    "Repository URL": (item.get("Repository URL") or item.get("仓库地址") or ""),
                    "Brief Introduction": brief_intro,
                    "Innovations": innovations,
                    "Basic Usage": basic_usage,
                    "Summary": summary,
                }

                for existing_repo in repos:
                    if existing_repo["full_name"] == repo_name:
                        full_entry["Repository URL"] = existing_repo.get("html_url", full_entry["Repository URL"])
                        break

                results[repo_name] = full_entry

    return results


def is_valid_summary(summary: str, language: str = "zh") -> bool:
    if not summary or not summary.strip():
        return False
    invalid_phrases = ["生成失败", "暂无AI总结", "429", "Copilot API限额已用尽", "RateLimitReached", "Not specified"]
    for phrase in invalid_phrases:
        if phrase in summary:
            return False

    common_english_templates = [
        r"Here'?s the summary",
        r"Here is the summary",
        r"Repository Name",
        r"Brief Introduction",
        r"Innovations",
        r"Basic Usage",
        r"Summary\s*:",
        r"Please summarize",
    ]

    common_chinese_templates = [
        r"仓库名称",
        r"简要介绍",
        r"创新点",
        r"简单用法",
        r"总结\s*[:：]",
        r"请对以下 GitHub 仓库进行内容总结",
    ]

    s_head = summary.strip()[:200]
    if language != "en":
        for p in common_english_templates:
            if re.search(p, s_head, flags=re.IGNORECASE):
                return False
    if language == "en":
        for p in common_chinese_templates:
            if re.search(p, s_head):
                return False

    full_text = summary.strip()
    if language == "en":
        patterns = [r"Summary\s*[:：]", r"Repository Name", r"Brief Introduction", r"Innovations"]
    else:
        patterns = [r"总结\s*[:：]", r"仓库名称", r"简要介绍", r"创新点"]

    missing = []
    for p in patterns:
        if not re.search(p, full_text, flags=re.IGNORECASE):
            missing.append(p)
    if missing:
        return False

    try:
        s = summary
        if language == "en":
            m = re.search(r"Brief Introduction\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", s, flags=re.IGNORECASE | re.S)
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                if len(intro_text) < 20:
                    return False
        else:
            m = re.search(r"简要介绍\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", s, flags=re.S)
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                if len(intro_text) < 10:
                    return False
    except Exception:
        pass

    return True


def build_repo_entry(repo: Dict, summary: Any) -> Dict:
    """Build a {full_name: entry} row for repo_summaries.json.

    The entry preserves ``__meta__`` (timestamp / hash / model) when present,
    so incremental selection in the next run can still work.
    """
    meta = None
    if isinstance(summary, dict):
        meta = summary.get("__meta__")
        # Brief Introduction: use LLM output, fallback to original description
        brief = summary.get("Brief Introduction", "") or ""
        if brief.lower().strip() in ("not specified.", "not specified", ""):
            brief = repo.get("description") or ""
        # Truncate to first paragraph only (Markdown single \n = space, causes layout issues)
        brief = brief.split("\n\n")[0].split("\n")[0].strip()
        entry = {
            "Repository Name": summary.get("Repository Name", repo.get("full_name")),
            "Repository URL": summary.get("Repository URL", repo.get("html_url")),
            "Brief Introduction": _clean_prompt_leak(brief),
            "Innovations": _clean_prompt_leak(summary.get("Innovations", "")),
            "Basic Usage": _clean_prompt_leak(summary.get("Basic Usage", "")),
            "Summary": _clean_prompt_leak(summary.get("Summary", "")),
        }
        entry["Repository URL"] = repo.get("html_url") or entry.get("Repository URL", "")
        # Preserve fields not surfaced to README but useful for next run:
        for k in ("Stars", "Forks", "updated", "language", "__last_error__"):
            if k in summary and k not in entry:
                entry[k] = summary[k]
        if meta:
            entry["__meta__"] = meta
        return entry

    return {
        "Repository Name": repo.get("full_name"),
        "Repository URL": repo.get("html_url"),
        "Brief Introduction": "",
        "Innovations": "",
        "Basic Usage": "",
        "Summary": summary or "",
    }


def select_repos_for_update(
    classified: Dict[str, List[Dict]],
    old_summaries: Dict[str, str],
    mode: str,
    language: str = "zh",
    description_lookup: Optional[Dict[str, str]] = None,
    refresh_after_days: int = 0,
) -> Dict[str, List[Dict]]:
    """Decide which repos actually need an LLM call this run.

    Modes:
      - 'force_all': every repo needs summarising (skip freshness checks).
      - 'all': every repo is re-examined, but hash-unchanged entries are reused.
      - 'missing_only': only entries without a valid summary are re-summarised.

    Args:
        classified: language -> [repo, ...] from the GitHub API
        old_summaries: full_name -> summary text or entry dict (legacy / quick lookup)
        mode: 'force_all' / 'all' / 'missing_only'
        language: 'en' or 'zh' (for content validation)
        description_lookup: full_name -> raw GitHub description. When provided,
            we can do a *content-hash* check: if the upstream description is
            unchanged, we keep the old summary even in 'all' mode (saves API
            budget). Pass None to fall back to plain text validation.
        refresh_after_days: if > 0, also re-summarize entries older than this
            many days (used for periodic refresh, 0 = never re-refresh).

    Returns:
        language -> [repo, ...] that still need summarising.
    """
    if mode == "force_all":
        return classified

    if mode not in ("missing_only", "all"):
        # Unknown mode => be conservative and only re-summarise missing ones.
        mode = "missing_only"

    if mode == "all" and not description_lookup:
        # Without per-repo descriptions we cannot hash-check, so legacy 'all'
        # really means 're-summarise everything'.
        return classified

    filtered: Dict[str, List[Dict]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = _repo_key(repo)
            entry = _resolve_entry(old_summaries, key)
            desc = (description_lookup or {}).get(key, "") if description_lookup else ""

            if mode == "missing_only":
                # Quick path: keep the legacy validation for callers that
                # don't pass per-repo descriptions.
                if description_lookup is None:
                    fallback = old_summaries.get(key, "")
                    if is_valid_summary(fallback, language):
                        continue
                    needs_update.append(repo)
                    continue

            # Hash-based freshness check.
            from scripts.core.json_store import is_entry_fresh
            if is_entry_fresh(entry, key, desc, refresh_after_days=refresh_after_days):
                continue

            needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def _resolve_entry(old_summaries: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """`old_summaries` may carry either a legacy str value or a rich entry dict.

    This helper returns the entry dict when available, else None.
    """
    if not key or key not in old_summaries:
        return None
    val = old_summaries.get(key)
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val:
        # Wrap legacy string summary so is_entry_fresh() can reason about it.
        return {
            "Brief Introduction": "",
            "Innovations": "",
            "Basic Usage": "",
            "Summary": val,
        }
    return None


def summarize_batch(
    repos: List[Dict],
    old_summaries: Dict[str, Any],
    summarize_func: Callable[[Dict], Optional[str]],
    update_mode: str,
    language: str,
    max_workers: int = 5,
    api_budget_tracker: Optional[Callable[[], bool]] = None,
) -> List[Any]:
    from scripts.ai.llm_caller import RateLimitAbort
    
    results: List[Any] = ["" for _ in repos]
    rate_limit_hit = False

    repos_with_prompts = []
    for repo in repos:
        repo_copy = dict(repo)
        repo_copy["prompt"] = generate_summarize_prompt(repo_copy, language)
        repos_with_prompts.append(repo_copy)

    # Track which repos were submitted vs deferred due to budget
    submitted_indices = []
    deferred_indices = []

    # Pre-filter repos based on budget before submitting to thread pool
    for idx, repo in enumerate(repos_with_prompts):
        if api_budget_tracker is not None and not api_budget_tracker():
            # Budget exhausted, defer remaining repos
            deferred_indices.append(idx)
            continue
        submitted_indices.append(idx)

    if deferred_indices:
        print(f"[BUDGET] API budget exhausted, deferring {len(deferred_indices)} repos for next run")

    # Handle deferred repos: preserve old summaries
    for idx in deferred_indices:
        repo = repos[idx]
        key = repo["full_name"]
        existing = old_summaries.get(key, {})
        if isinstance(existing, dict) and existing.get("Summary"):
            results[idx] = existing
        else:
            results[idx] = {
                "Repository Name": key,
                "Repository URL": repo.get("html_url") or "",
                "Brief Introduction": "",
                "Innovations": "",
                "Basic Usage": "",
                "Summary": "(deferred: API budget exhausted)",
            }

    # Process submitted repos concurrently
    if not submitted_indices:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(summarize_func, repos_with_prompts[idx]): idx for idx in submitted_indices}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            repo = repos[idx]
            try:
                existing_entry = old_summaries.get(repo["full_name"], {})
                # Extract Summary string from dict entry for validation
                if isinstance(existing_entry, dict):
                    existing_summary_str = existing_entry.get("Summary", "")
                else:
                    existing_summary_str = existing_entry or ""
                
                reuse_existing = (update_mode == "missing_only") and is_valid_summary(existing_summary_str, language)
                if reuse_existing:
                    # Return full dict entry to preserve __meta__ metadata
                    summary = existing_entry if isinstance(existing_entry, dict) else existing_summary_str
                else:
                    summary = future.result()
                    if summary is None:
                        api_name = summarize_func.__name__.replace("_summarize", "").upper()
                        summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败或429")
                
                # Only call strip() on strings
                if isinstance(summary, str):
                    print(f"[DEBUG] [repo]: {repo['full_name']} | [AI summary]: {repr(summary.strip())}")
                else:
                    summary_text = summary.get("Summary", "") if isinstance(summary, dict) else str(summary)
                    print(f"[DEBUG] [repo]: {repo['full_name']} | [AI summary]: {repr(summary_text)}")
            except RateLimitAbort as exc:
                print(f"[RATE_LIMIT] {repo['full_name']} 触发速率限制: {exc}")
                print(f"[RATE_LIMIT] 主动停止后续请求，保存已有结果")
                rate_limit_hit = True
                # Preserve old summary for this repo
                existing = old_summaries.get(repo["full_name"], {})
                if isinstance(existing, dict) and existing.get("Summary"):
                    results[idx] = existing
                else:
                    results[idx] = {
                        "Repository Name": repo["full_name"],
                        "Repository URL": repo.get("html_url") or "",
                        "Brief Introduction": "",
                        "Innovations": "",
                        "Basic Usage": "",
                        "Summary": "(deferred: rate limit reached)",
                    }
                # Cancel remaining futures
                for f in future_to_idx:
                    f.cancel()
                break
            except Exception as exc:
                print(f"{repo['full_name']} 线程异常: {exc}")
                api_name = summarize_func.__name__.replace("_summarize", "").upper()
                summary = old_summaries.get(repo["full_name"], f"{api_name} API生成失败")
            results[idx] = summary if summary is not None else "*暂无AI总结*"
    
    # Handle remaining repos if rate limit was hit
    if rate_limit_hit:
        for idx in submitted_indices:
            if results[idx] == "":
                repo = repos[idx]
                key = repo["full_name"]
                existing = old_summaries.get(key, {})
                if isinstance(existing, dict) and existing.get("Summary"):
                    results[idx] = existing
                else:
                    results[idx] = {
                        "Repository Name": key,
                        "Repository URL": repo.get("html_url") or "",
                        "Brief Introduction": "",
                        "Innovations": "",
                        "Basic Usage": "",
                        "Summary": "(deferred: rate limit reached)",
                    }
    
    return results


def summarize_batch_combined(
    repos: List[Dict],
    old_summaries: Dict[str, Any],
    summarize_func: Callable[[Dict], Optional[str]],
    update_mode: str,
    language: str,
    batch_size: int = 5,
    batch_num: int = 1,
    api_budget_tracker: Optional[Callable[[], bool]] = None,
    description_lookup: Optional[Dict[str, str]] = None,
    model_name: str = "unknown",
) -> List[Any]:
    """Summarise a batch of repos with per-repo fallback and budget control.

    New behaviour vs. the original implementation:
      * When a combined batch succeeds but the parse misses some repos, the
        missing repos are re-tried one-by-one (single-repo call) instead of
        being silently written as empty. This is the fix for the 11-Unknown
        regression we saw on 2026-06-18.
      * `api_budget_tracker` is an optional callable that returns False when
        the run has exhausted its per-run API budget. When False, we stop
        scheduling new batches and preserve the existing summary (if any).
      * Successful entries are stamped with metadata (last_summarized_at,
        description_hash, model, source) so future runs can do hash-based
        incremental selection.

    `api_budget_tracker` signature: () -> bool (True = still has budget)
    """
    from scripts.ai.llm_caller import RateLimitAbort

    results: List[Any] = [{} for _ in repos]

    repos_need_call: List[Dict] = []
    repos_indices: List[int] = []

    for idx, repo in enumerate(repos):
        key = repo["full_name"]
        existing_summary = old_summaries.get(key, {})
        desc = (description_lookup or {}).get(key, "") if description_lookup else ""

        # Hash-based reuse: skip repos whose upstream description is
        # unchanged AND whose existing entry is well-formed. Disabled when
        # update_mode == 'force_all' so the user can force a refresh.
        if update_mode != "force_all" and (update_mode == "missing_only" or description_lookup is not None):
            from scripts.core.json_store import is_entry_fresh, make_metadata
            if isinstance(existing_summary, dict):
                entry = existing_summary
            elif isinstance(existing_summary, str) and existing_summary:
                entry = {
                    "Brief Introduction": "",
                    "Innovations": "",
                    "Basic Usage": "",
                    "Summary": existing_summary,
                }
            else:
                entry = None
            if entry is not None and is_entry_fresh(entry, key, desc, refresh_after_days=0):
                # Backfill __meta__ on legacy entries so the next run can do
                # accurate hash-based checks and track freshness.
                if "__meta__" not in entry or not entry["__meta__"].get("last_summarized_at"):
                    entry["__meta__"] = make_metadata(
                        full_name=key,
                        description=desc,
                        model=entry.get("__meta__", {}).get("summary_model", "legacy"),
                        source="backfill",
                        attempts=0,
                    )
                results[idx] = entry
                print(f"[REUSE] repo: {key} | fresh (hash={entry.get('description_hash', '?')[:8]})")
                continue

        # Legacy compatibility: legacy string summary without metadata.
        if isinstance(existing_summary, dict) and existing_summary.get("Summary"):
            if update_mode == "missing_only":
                results[idx] = existing_summary
                print(f"[REUSE] repo: {key} | existing summary (dict)")
                continue
        elif isinstance(existing_summary, str) and existing_summary:
            if update_mode == "missing_only":
                results[idx] = {
                    "Repository Name": key,
                    "Repository URL": repo.get("html_url") or "",
                    "Brief Introduction": "",
                    "Innovations": "",
                    "Basic Usage": "",
                    "Summary": existing_summary,
                }
                print(f"[REUSE] repo: {key} | existing summary (legacy str)")
                continue

        repos_need_call.append(repo)
        repos_indices.append(idx)

    if not repos_need_call:
        return results

    # Process in sub-batches (each sub-batch is one LLM call).
    for i in range(0, len(repos_need_call), batch_size):
        if api_budget_tracker is not None and not api_budget_tracker():
            print(f"[BUDGET] API budget exhausted, preserving {len(repos_need_call) - i} repos for next run")
            for j, repo in enumerate(repos_need_call[i:]):
                idx = repos_indices[i + j]
                key = repo["full_name"]
                # Preserve any legacy summary we already have.
                existing = old_summaries.get(key, {})
                if isinstance(existing, dict) and existing.get("Summary"):
                    results[idx] = existing
                else:
                    results[idx] = {
                        "Repository Name": key,
                        "Repository URL": repo.get("html_url") or "",
                        "Brief Introduction": "",
                        "Innovations": "",
                        "Basic Usage": "",
                        "Summary": "(deferred: API budget exhausted)",
                    }
            break

        batch = repos_need_call[i : i + batch_size]
        indices = repos_indices[i : i + batch_size]
        print(f"[COMBINED] Processing batch {batch_num}, {len(batch)} repos...")

        combined_prompt = generate_combined_summarize_prompt(batch, language)
        repo_with_prompt = {"prompt": combined_prompt, "repos": [r["full_name"] for r in batch]}
        print(f"[DEBUG] Batch {batch_num} prompt length: {len(combined_prompt)} chars, repos: {[r['full_name'] for r in batch]}", flush=True)
        print(f"[DEBUG] Batch {batch_num} calling summarize_func...", flush=True)

        parsed_results: Dict[str, Dict] = {}
        try:
            response_text = summarize_func(repo_with_prompt)
            if response_text:
                parsed_results = parse_combined_summaries(response_text, batch)
            else:
                print(f"[DEBUG] Batch {batch_num}: summarize_func returned None/empty", flush=True)
        except RateLimitAbort as exc:
            print(f"[RATE_LIMIT] Batch {batch_num} 触发速率限制: {exc}", flush=True)
            print(f"[RATE_LIMIT] 主动停止后续批次，保存已有结果", flush=True)
            # Preserve old summaries for remaining repos in this batch and all subsequent batches
            for j, repo in enumerate(repos_need_call[i:]):
                idx_to_preserve = repos_indices[i + j]
                key = repo["full_name"]
                existing = old_summaries.get(key, {})
                if isinstance(existing, dict) and existing.get("Summary"):
                    results[idx_to_preserve] = existing
                else:
                    results[idx_to_preserve] = _placeholder_entry(repo, old_summaries, "rate limit reached")
            break
        except Exception as exc:
            import traceback
            print(f"[ERROR] Batch {batch_num} exception: {exc}", flush=True)
            print(f"[ERROR] Exception details: {traceback.format_exc()}", flush=True)

        # Per-repo fallback: any repo that the combined parse missed or
        # produced an empty Summary for gets a single-repo retry.
        missing: List[tuple] = []  # (idx, repo, attempts)
        for idx, repo in zip(indices, batch):
            key = repo["full_name"]
            entry = parsed_results.get(key) or {}
            summary_text = (entry.get("Summary") or "").strip()
            brief = (entry.get("Brief Introduction") or "").strip()
            if not summary_text or not brief or summary_text == "Not specified.":
                missing.append((idx, repo, 1))

        if missing:
            print(f"[FALLBACK] Combined parse missed {len(missing)}/{len(batch)} repos, retrying individually...")
            for idx, repo, attempt in missing:
                if api_budget_tracker is not None and not api_budget_tracker():
                    print(f"[BUDGET] skipping single-repo retry for {repo['full_name']}")
                    results[idx] = _placeholder_entry(repo, old_summaries, "API budget exhausted")
                    continue
                try:
                    single_prompt = generate_summarize_prompt(repo, language)
                    single_text = summarize_func({"prompt": single_prompt, "full_name": repo["full_name"]})
                    if single_text:
                        single_dict = _parse_single_repo_summary(single_text, repo)
                        if single_dict.get("Summary") and single_dict.get("Summary") != "Not specified.":
                            parsed_results[repo["full_name"]] = single_dict
                            print(f"[FALLBACK-OK] {repo['full_name']}: single-repo retry succeeded")
                            continue
                except RateLimitAbort as exc:
                    print(f"[RATE_LIMIT] Fallback retry 触发速率限制: {exc}")
                    print(f"[RATE_LIMIT] 停止所有 fallback 重试，保存已有结果")
                    # Preserve old summary for this and remaining missing repos
                    for remaining_idx, remaining_repo, _ in missing[missing.index((idx, repo, attempt)):]:
                        results[remaining_idx] = _placeholder_entry(remaining_repo, old_summaries, "rate limit reached")
                    break
                except Exception as e:
                    print(f"[FALLBACK-ERR] {repo['full_name']}: {e}")
                # Still missing after fallback: preserve old or placeholder.
                results[idx] = _placeholder_entry(repo, old_summaries, "Combined+single LLM returned empty")

        # Stamp metadata and finalise results.
        from datetime import datetime, timezone
        from scripts.core.json_store import make_metadata
        for idx, repo in zip(indices, batch):
            key = repo["full_name"]
            entry = parsed_results.get(key) or {}
            if not entry.get("Summary") or entry.get("Summary") == "Not specified.":
                # Final fallback: keep old summary if any, else placeholder.
                entry = _placeholder_entry(repo, old_summaries, "LLM returned empty after retry")
            desc = (description_lookup or {}).get(key, repo.get("description") or "")
            entry.setdefault("Repository Name", key)
            entry.setdefault("Repository URL", repo.get("html_url") or "")
            entry["__meta__"] = make_metadata(
                full_name=key,
                description=desc,
                model=model_name,
                source="ai",
                attempts=1,
            )
            results[idx] = entry

    return results


def _placeholder_entry(repo: Dict, old_summaries: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Build a placeholder entry while preserving any previously-good summary.

    The placeholder Summary keeps the previous content if it was valid so the
    next run can re-try without losing data. The reason is recorded in
    `__last_error__` (consumed by diagnostics, not the README renderer).
    """
    key = repo.get("full_name") or ""
    existing = old_summaries.get(key) if isinstance(old_summaries, dict) else None
    if isinstance(existing, dict) and (existing.get("Summary") or "").strip():
        out = dict(existing)
        out["__last_error__"] = reason
        return out
    return {
        "Repository Name": key,
        "Repository URL": repo.get("html_url") or "",
        "Brief Introduction": "",
        "Innovations": "",
        "Basic Usage": "",
        "Summary": "",
        "__last_error__": reason,
    }


def _parse_single_repo_summary(text: str, repo: Dict) -> Dict[str, Any]:
    """Best-effort parse of a single-repo summary response (non-JSON)."""
    import re
    if not text:
        return {}
    out = {
        "Repository Name": repo.get("full_name") or "",
        "Repository URL": repo.get("html_url") or "",
    }
    # Match either English or Chinese labels.
    pairs = [
        (r"Brief Introduction\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Brief Introduction"),
        (r"简要介绍\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Brief Introduction"),
        (r"Innovations\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Innovations"),
        (r"创新点\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Innovations"),
        (r"Basic Usage\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Basic Usage"),
        (r"简单用法\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Basic Usage"),
        (r"Summary\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Summary"),
        (r"总结\s*[:：]\s*(.+?)(?:\n\s*\d+\.|\n\n|$)", "Summary"),
    ]
    for pat, field in pairs:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            # Avoid overwriting a populated field with a short junk match.
            if val and val != "Not specified.":
                out[field] = val
    return out


def sort_repos_by_validity(
    repos: List[Dict],
    old_summaries: Dict[str, str],
    language: str = "zh",
) -> List[Dict]:
    try:
        return sorted(repos, key=lambda r: is_valid_summary(old_summaries.get(r.get("full_name") or "", ""), language))
    except Exception:
        return repos


def get_summarize_func(
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
    config: Dict[str, Any],
    throttle: Any,
    request_timeout: float,
    request_retry_delay: float,
    retry_attempts: int,
    api_call_counter: Callable,
):
    from scripts.ai.api_clients import create_summarize_func

    return create_summarize_func(
        model_choice=model_choice,
        github_token=github_token,
        openrouter_api_key=openrouter_api_key,
        gemini_api_key=gemini_api_key,
        modelscope_api_key=modelscope_api_key,
        default_copilot_model=default_copilot_model,
        default_openrouter_model=default_openrouter_model,
        default_gemini_model=default_gemini_model,
        default_modelscope_model=default_modelscope_model,
        language=language,
        config=config,
        throttle=throttle,
        request_timeout=request_timeout,
        request_retry_delay=request_retry_delay,
        retry_attempts=retry_attempts,
        api_call_counter=api_call_counter,
    )
