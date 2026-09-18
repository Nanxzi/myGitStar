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
    r"\*\*\s*\(.*?若仓库确实没有可用用法.*?\)\s*\n?",
    r"\*\*\s*\(Write 'Not specified' ONLY if.*?\)\s*\n?",
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


# Editorial commentary the LLM sometimes appends after a field value,
# e.g. "*If the repository does not contain explicit CLI instructions...*".
_EDITORIAL_PATTERNS = [
    r"^\s*\*?\s*If (?:the|a) repository",
    r"^\s*\*?\s*If none",
    r"^\s*\*?\s*Note[s]?:",
    r"^\s*\*?\s*Note that",
    r"^\s*\*?\s*Please note",
    r"^\s*\*?This (?:reflects|is|summar)",
    r"^\s*\*?\(?As (?:an |the )?(?:AI|assistant|model)",
]


def _strip_code_fences(text: str) -> str:
    """Unwrap markdown ```...``` fences LLMs sometimes wrap a value in.

    A fenced value (notably Basic Usage) breaks the README's list structure
    once embedded, so we drop the opening/closing fence lines and any stray
    triple-backticks. Single-backtick inline code is preserved.
    """
    if not text:
        return text
    s = text.strip()
    # Drop an opening fence line (``` optionally followed by a language tag).
    s = re.sub(r"^```[^\n]*\n", "", s)
    # Drop any closing fence lines (with optional trailing whitespace).
    s = re.sub(r"\n```[ \t]*", "\n", s)
    s = re.sub(r"[ \t]*```$", "", s)
    s = re.sub(r"^```[ \t]*", "", s)
    # Remove any leftover stray triple-backticks anywhere.
    s = s.replace("```", "")
    return s.strip()


def _strip_editorial_commentary(text: str) -> str:
    """Drop standalone/ trailing editorial notes the LLM adds after a value."""
    if not text:
        return text
    kept = []
    for line in text.split("\n"):
        if any(re.match(p, line, flags=re.IGNORECASE) for p in _EDITORIAL_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


# Field labels the LLM sometimes re-emits *inside* a single field value to
# chain several structured fields together (e.g. Innovations bleeding into
# "Basic Usage:** ..." Summary:** ..."). We strip those labels AND everything
# that follows them so one field never carries another field's value.
_EMBEDDED_FIELD_LABELS = r"(?:Repository (?:URL|Description)|Basic Usage|Summary|仓库地址|仓库描述|简单用法|总结)"


def _strip_embedded_field_labels(text: str) -> str:
    """Remove embedded downstream field labels inside a value, plus leftovers.

    When the LLM appends something like ``  \\n**Basic Usage:** <usage>`` (or
    ``  **Summary:** <text>``) onto the end of an earlier field, we drop the
    label anchor and everything after it. A label must be followed by a colon
    and at least one non-trivial char so we do not clip a value that merely
    mentions the word ``Summary`` in prose.
    """
    if not text:
        return text
    # Anchor on an embedded field label that carries a colon and real content.
    pattern = re.compile(
        rf"(?:\n\s*)?\**\s*{_EMBEDDED_FIELD_LABELS}\s*[:：]\s*\S",
        flags=re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return text
    return text[: m.start()].rstrip("\n ") or text[: m.start()].strip()


def _strip_trailing_asterisk(text: str) -> str:
    """Remove a dangling asterisk markdown marker that the LLM sometimes leaves
    at the end of a field value (e.g. ``...\n*``). Only removes an *isolated*
    marker line, never inline content or trailing emphasis.
    """
    if not text:
        return text
    # Drop a trailing line that consists only of asterisks (preceded by a
    # newline or at the start of the string). "word**" emphasis is preserved.
    text = re.sub(r"(?:\n\s*|\A)\*{1,3}\s*\Z", "", text)
    return text.rstrip()


def _clean_field_value(text: str) -> str:
    """Full cleaning for a single summary field value.

    Combines prompt-leak removal with code-fence unwrapping, embedded-label
    stripping, trailing-asterisk removal and editorial-commentary removal so
    stored/rendered values never break markdown.
    """
    if not text:
        return text
    text = _clean_prompt_leak(text)
    text = _strip_code_fences(text)
    text = _strip_embedded_field_labels(text)
    text = _strip_editorial_commentary(text)
    text = _strip_trailing_asterisk(text)
    return text


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
            f"4. **简单用法：** （基于 README 内容，给出一段简短的使用说明：怎么用、怎么配置、怎么启动，控制在一两句话、50字以内，不要写成单行命令。若仓库确实没有可用用法（如纯数据集、纯论文、纯资料）才写 Not specified。禁止 import 语句、禁止 ``` 围栏、禁止完整代码块。例如：“通过 HTTPS 服务访问、默认端口 8080、可用 Docker 启动”，而非 `watermark embed input.jpg output.jpg`。）\n"
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
            f"4. **Basic Usage:** (Based on README, give a short description of how to use/configure/start it, keeping it to one or two sentences within 50 words; do NOT write a one-line command. Write 'Not specified' ONLY if the repo truly has no applicable usage, e.g. a pure dataset/paper/material repo. No import statements, no ``` fences, no full code blocks. E.g. `served over HTTPS on :8080, launchable via Docker`, rather than `watermark embed input.jpg output.jpg`.)\n"
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
            "- Basic Usage: 基于 README 内容，给出一段简短的使用说明：怎么用、怎么配置、怎么启动，控制在一两句话、50字以内，不要写成单行命令。仅当仓库确实无可用用法（如纯数据集、纯论文、纯资料）才写 Not specified。禁止 import 语句、禁止 ``` 围栏、禁止完整代码块。例如：“通过 HTTPS 服务访问、默认端口 8080、可用 Docker 启动”，而非 `watermark embed input.jpg output.jpg`\n"
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
            "- Basic Usage: based on README, give a short description of how to use/configure/start it, keeping it to one or two sentences within 50 words; do NOT write a one-line command. Write 'Not specified' ONLY if the repo truly has no applicable usage, e.g. a pure dataset/paper/material repo. No import statements, no ``` fences, no full code blocks. E.g. `served over HTTPS on :8080, launchable via Docker`, rather than `watermark embed input.jpg output.jpg`\n"
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

                # Clean prompt leak artifacts, code fences and editorial notes
                brief_intro = _clean_field_value(brief_intro)
                innovations = _clean_field_value(innovations)
                basic_usage = _clean_field_value(basic_usage)
                summary = _clean_field_value(summary)

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


# Placeholder values used for Basic Usage when a repo genuinely/nominally
# has no applicable usage. A repo whose Basic Usage carries one of these is
# treated by ``missing_only`` mode as "needs update" so a relaxed prompt
# (descriptive usage, not one-line command) can be given another pass.
_BASIC_USAGE_MISSING = {
    "not specified", "not specified.", "n/a", "na",
    "无", "暂无", "未指定", "--", "-", "—", "",
}


def is_basic_usage_missing(entry) -> bool:
    """Whether a stored summary entry lacks a usable Basic Usage value.

    Unlike ``is_valid_summary`` (which only requires the overall summary to be
    structurally sound), this targets just the Basic Usage field. It returns
    True when the field is absent, empty, or a placeholder such as
    'Not specified'/'N/A', signalling that ``missing_only`` should re-run this
    repo so the relaxed prompt can fill it with a short descriptive usage.
    """
    if not isinstance(entry, dict):
        return True
    val = str(entry.get("Basic Usage") or entry.get("简单用法") or "").strip()
    return val.lower() in _BASIC_USAGE_MISSING


def is_valid_summary(summary: str, language: str = "zh") -> bool:
    """Check whether a raw LLM summary string is structurally valid.

    The key change (2026-08-20): 'Not specified' is now a LEGITIMATE
    placeholder (the repo genuinely has no such content), NOT a signal
    that the LLM failed.  We only invalidate when *all* key fields are
    'Not specified' (total LLM failure) or when severe error signals are
    present.
    """
    if not summary or not summary.strip():
        return False

    # Severe error signals — these always indicate an LLM/API failure.
    severe_error_phrases = [
        "生成失败", "暂无AI总结", "429",
        "Copilot API限额已用尽", "RateLimitReached",
        "(deferred",
    ]
    for phrase in severe_error_phrases:
        if phrase in summary:
            return False

    # ---- Structured check 1: JSON-formatted summary ----
    # If the LLM returned a JSON string, check if ALL key fields are
    # 'Not specified' (total failure) vs. partial placeholders (valid).
    try:
        import json
        parsed = json.loads(summary.strip())
        if isinstance(parsed, dict):
            required = ["Brief Introduction", "Innovations", "Summary"]
            all_not_specified = True
            has_any_content = False
            for f in required:
                v = str(parsed.get(f, "") or "").strip()
                if not v:
                    continue
                if v.lower() not in ("not specified.", "not specified"):
                    has_any_content = True
                    all_not_specified = False
            # If every required field is either empty or "Not specified",
            # the LLM totally failed — mark as invalid.
            if all_not_specified and not has_any_content:
                return False
            # JSON parsed successfully and has some valid content — accept.
            return True
    except (json.JSONDecodeError, TypeError):
        pass  # Not a JSON string, fall through to text-based checks.

    # ---- Text-based template checks (non-JSON summaries) ----
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

    # Length check: ensure the generated content has enough text
    # (not just a list of "Not specified" placeholders).
    # Use flexible field-end patterns that match:
    #   - numbered list items (\n 1. or \n 2.)
    #   - double newlines (\n\n)
    #   - next field label (e.g. "\nInnovations:")
    #   - end of string ($)
    try:
        s = summary
        if language == "en":
            # Build flexible field-end pattern for English
            next_labels = r"(?:Repository Name|Brief Introduction|Innovations|Basic Usage|Summary)"
            field_end = rf"(?:\n\s*\d+\.|\n\n|\n\s*{next_labels}\s*[:：]|$)"

            m = re.search(
                rf"Brief Introduction\s*[:：]\s*(.+?){field_end}",
                s, flags=re.IGNORECASE | re.S
            )
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                # Check if intro is ALL "Not specified" — total failure
                if intro_text.lower() in ("not specified.", "not specified", ""):
                    # Check other fields too
                    all_fields_empty = True
                    for field_name in ["Innovations", "Summary"]:
                        fm = re.search(
                            rf"{field_name}\s*[:：]\s*(.+?){field_end}",
                            s, flags=re.IGNORECASE | re.S
                        )
                        if fm:
                            ft = fm.group(1).strip()
                            ft_clean = re.sub(r"\*|\*\*|`|\\n", "", ft).strip()
                            if ft_clean.lower() not in ("not specified.", "not specified", ""):
                                all_fields_empty = False
                                break
                    if all_fields_empty:
                        return False  # All fields are "Not specified" — LLM failed
                # If intro has real content but < 20 chars, still reject
                elif len(intro_text) < 20:
                    return False
        else:
            # Build flexible field-end pattern for Chinese
            next_labels_zh = r"(?:仓库名称|简要介绍|创新点|简单用法|总结)"
            field_end_zh = rf"(?:\n\s*\d+\.|\n\n|\n\s*{next_labels_zh}\s*[:：]|$)"

            m = re.search(
                rf"简要介绍\s*[:：]\s*(.+?){field_end_zh}",
                s, flags=re.S
            )
            if m:
                intro = m.group(1).strip()
                intro_text = re.sub(r"\*|\*\*|`|\\n", "", intro).strip()
                if intro_text.lower() in ("not specified.", "not specified", ""):
                    all_fields_empty = True
                    for field_name in ["创新点", "总结"]:
                        fm = re.search(
                            rf"{field_name}\s*[:：]\s*(.+?){field_end_zh}",
                            s, flags=re.S
                        )
                        if fm:
                            ft = fm.group(1).strip()
                            ft_clean = re.sub(r"\*|\*\*|`|\\n", "", ft).strip()
                            if ft_clean.lower() not in ("not specified.", "not specified", ""):
                                all_fields_empty = False
                                break
                    if all_fields_empty:
                        return False
                elif len(intro_text) < 10:
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
            "Brief Introduction": _clean_field_value(brief),
            "Innovations": _clean_field_value(summary.get("Innovations", "")),
            "Basic Usage": _clean_field_value(summary.get("Basic Usage", "")),
            "Summary": _clean_field_value(summary.get("Summary", "")),
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
                    if is_valid_summary(fallback, language) and not is_basic_usage_missing(entry):
                        continue
                    needs_update.append(repo)
                    continue

            # Hash-based freshness check.
            from scripts.core.json_store import is_entry_fresh
            if is_entry_fresh(entry, key, desc, refresh_after_days=refresh_after_days):
                # Keep a fresh entry only if it carries a usable Basic Usage;
                # otherwise re-run so the relaxed prompt (descriptive usage,
                # not a one-line command) can fill this field.
                if not is_basic_usage_missing(entry):
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
    rate_limit_exc: Optional[RateLimitAbort] = None

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
                
                reuse_existing = (
                    update_mode == "missing_only"
                    and is_valid_summary(existing_summary_str, language)
                    and not is_basic_usage_missing(existing_entry)
                )
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
                # Fill placeholder for the repo that triggered the rate limit
                # and cancel queued (not yet started) futures, but keep
                # draining in-flight ones so their completed work is still
                # persisted instead of being thrown away.
                print(f"[RATE_LIMIT] {repo['full_name']} 触发速率限制: {exc}")
                print(f"[RATE_LIMIT] 取消排队请求，等待进行中的请求完成后保存结果")
                rate_limit_hit = True
                if rate_limit_exc is None:
                    rate_limit_exc = exc
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
                # Cancel queued futures only; running ones finish and their
                # results are harvested by the loop below.
                for f in future_to_idx:
                    f.cancel()
                continue
            except concurrent.futures.CancelledError:
                # Cancelled while queued after the rate limit; the post-loop
                # block fills its placeholder from the old summary.
                continue
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

    # Re-raise so the main loop knows to stop processing subsequent batches.
    # Already-processed results above are preserved in the `results` list
    # and attached to the exception so the caller can still persist them.
    if rate_limit_hit and rate_limit_exc is not None:
        rate_limit_exc.results = results
        raise rate_limit_exc

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
    backoff_base_days: float = 1.0,
    backoff_max_days: float = 8.0,
    rate_limit_batch_retries: int = 2,
    rate_limit_retry_wait: float = 60.0,
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
        # NOTE: entries whose Summary is the legacy "(deferred ...)" poison
        # string must NOT be reused — they need a real LLM call.
        if isinstance(existing_summary, dict) and existing_summary.get("Summary"):
            summary_txt = str(existing_summary.get("Summary") or "").strip()
            if not summary_txt.startswith("(deferred"):
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
                # Record the deferral in __last_error__ (NOT in Summary —
                # writing it into Summary used to poison the reuse check
                # and starve these repos forever).
                results[idx] = _placeholder_entry(repo, old_summaries, "deferred: API budget exhausted")
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
            # Transient rate limits (e.g. per-minute caps) often clear within
            # a minute, so before aborting the whole run we retry the same
            # batch a bounded number of times with a linearly increasing
            # wait. RateLimitAbort raised *inside* summarize_func (RPD
            # exhaustion, consecutive-429 threshold) bypasses these retries.
            response_text = None
            for rate_limit_attempt in range(rate_limit_batch_retries + 1):
                response_text = summarize_func(repo_with_prompt)
                # Check if response is a 429 error marker (dict with __error__ = "429")
                if isinstance(response_text, dict) and response_text.get("__error__") == "429":
                    if rate_limit_attempt < rate_limit_batch_retries:
                        wait = rate_limit_retry_wait * (rate_limit_attempt + 1)
                        print(
                            f"[RATE_LIMIT] Batch {batch_num}: 429 after fallback chain; "
                            f"sleeping {wait:.0f}s then retrying batch "
                            f"({rate_limit_attempt + 1}/{rate_limit_batch_retries})",
                            flush=True,
                        )
                        time.sleep(wait)
                        continue
                    print(f"[RATE_LIMIT] Batch {batch_num}: received 429 error marker", flush=True)
                    raise RateLimitAbort(f"Batch {batch_num} received 429 rate limit error")
                if isinstance(response_text, dict):
                    print(f"[DEBUG] Batch {batch_num}: summarize_func returned unexpected dict", flush=True)
                    response_text = None
                break
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
            # Re-raise so the main loop can break out of subsequent batches
            # instead of retrying the same rate-limited API. Attach the
            # results list so already-completed repos are not lost.
            exc.results = results
            raise
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
            print(f"[FALLBACK] Combined parse missed {len(missing)}/{len(batch)} repos, retrying individually with delay...")
            for i_missing, (idx, repo, attempt) in enumerate(missing):
                # 非首个重试时，增加缓冲时间避免连续请求触发 429
                if i_missing > 0:
                    fallback_delay = 3.0  # 每个单独重试之间间隔 3 秒
                    print(f"[FALLBACK] Waiting {fallback_delay}s before next single-repo retry...")
                    time.sleep(fallback_delay)
                if api_budget_tracker is not None and not api_budget_tracker():
                    print(f"[BUDGET] skipping single-repo retry for {repo['full_name']}")
                    results[idx] = _placeholder_entry(repo, old_summaries, "API budget exhausted")
                    continue
                try:
                    single_prompt = generate_summarize_prompt(repo, language)
                    single_text = summarize_func({"prompt": single_prompt, "full_name": repo["full_name"]})
                    if isinstance(single_text, dict) and single_text.get("__error__") == "429":
                        # Rate-limit marker: stop the retry loop instead of
                        # letting the dict flow into the string parser below.
                        raise RateLimitAbort(
                            f"Single-repo retry for {repo['full_name']} received 429 rate limit error"
                        )
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
                    # Re-raise so the main loop can break out. Attach
                    # results so already-completed repos are not lost.
                    exc.results = results
                    raise
                except Exception as e:
                    print(f"[FALLBACK-ERR] {repo['full_name']}: {e}")
                # Still missing after fallback: preserve old or placeholder.
                results[idx] = _placeholder_entry(repo, old_summaries, "Combined+single LLM returned empty")

        # Stamp metadata and finalise results.
        from datetime import datetime, timezone
        from scripts.core.json_store import make_metadata, make_error_meta
        for idx, repo in zip(indices, batch):
            key = repo["full_name"]
            entry = parsed_results.get(key) or {}
            if not entry.get("Summary") or entry.get("Summary") == "Not specified.":
                # Final fallback: keep old summary if any, else placeholder.
                entry = _placeholder_entry(repo, old_summaries, "LLM returned empty after retry")
            desc = (description_lookup or {}).get(key, repo.get("description") or "")
            entry.setdefault("Repository Name", key)
            entry.setdefault("Repository URL", repo.get("html_url") or "")
            failed = bool(entry.get("__last_error__")) or not (entry.get("Summary") or "").strip()
            if failed:
                # Failed attempt: record error metadata with an exponential
                # backoff window instead of stamping a fake success.
                prev_attempts = 0
                prev_meta = entry.get("__meta__") if isinstance(entry.get("__meta__"), dict) else None
                if isinstance(prev_meta, dict):
                    try:
                        prev_attempts = int(prev_meta.get("summary_attempts") or 0)
                    except Exception:
                        prev_attempts = 0
                entry["__meta__"] = make_error_meta(
                    full_name=key,
                    description=desc,
                    model=model_name,
                    reason=entry.get("__last_error__") or "llm_empty",
                    attempts=prev_attempts + 1,
                    backoff_base_days=backoff_base_days,
                    backoff_max_days=backoff_max_days,
                )
            else:
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
    out = {
        "Repository Name": key,
        "Repository URL": repo.get("html_url") or "",
        "Brief Introduction": "",
        "Innovations": "",
        "Basic Usage": "",
        "Summary": "",
        "__last_error__": reason,
    }
    # Preserve the attempt history even when content fields are empty, so
    # the error-meta stamping can increment summary_attempts across runs.
    if isinstance(existing, dict) and isinstance(existing.get("__meta__"), dict):
        out["__meta__"] = existing["__meta__"]
    return out


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
            val = _clean_field_value(m.group(1))
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
