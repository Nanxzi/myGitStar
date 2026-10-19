import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Route legacy call sites through the newer cleaner (also unwraps ``` fences
# and strips editorial commentary) so existing dirty summaries render clean.
from scripts.summary.summarize_helpers import _clean_field_value as _clean_prompt_leak
from scripts.summary.summarize_helpers import normalize_placeholder


LANG_ICONS = {
    "Python": "🐍",
    "JavaScript": "🟨",
    "TypeScript": "🔷",
    "Java": "☕",
    "Go": "🐹",
    "Rust": "🦀",
    "C++": "⚡",
    "C": "🔧",
    "C#": "💜",
    "PHP": "🐘",
    "Ruby": "💎",
    "Swift": "🐦",
    "Kotlin": "🅺",
    "Dart": "🎯",
    "Shell": "🐚",
    "HTML": "🌐",
    "CSS": "🎨",
    "Vue": "💚",
    "React": "⚛️",
    "Other": "📦",
}


def classify_by_language(repos: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    classified: Dict[str, List[Dict[str, Any]]] = {}
    for repo in repos:
        lang = repo.get("language") or "Other"
        classified.setdefault(lang, []).append(repo)
    return classified


def github_anchor(text: str) -> str:
    anchor = text.strip().lower()
    anchor = re.sub(r"[\s]+", "-", anchor)
    anchor = re.sub(r"[^\w\u4e00-\u9fa5-]", "", anchor)
    return anchor


def _get_lang_icon(lang: str) -> str:
    return LANG_ICONS.get(lang, "📝")


def _format_updated_at(updated_at: str) -> str:
    if not updated_at:
        return ""
    try:
        dt = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(updated_at)[:10]


def build_repo_entry_lines(
    full_name: str,
    url: str,
    stars: int,
    forks: int,
    updated_at: str,
    brief: str,
    innovations: str,
    basic_usage: str,
    summary_text: str,
    language: str,
) -> List[str]:
    """Render one repo entry: heading, stat line, lead paragraph, detail bullets.

    Two deliberate choices, all about making the output readable:
    - the 'Repository Name' line is gone (it duplicates the heading);
    - the numbered pseudo-form becomes a lead paragraph plus bullets.

    Placeholders stay visible (dropping the field hides information) but are
    normalised to one canonical string, so 'N/A' / 'Not provided.' / 'Not
    specified (README content not provided)' all read the same.  Only a value
    that is an empty string is skipped, and '*No AI summary available*' is used
    only when every field is empty (i.e. no summary data exists at all).
    """
    is_en = language == "en"
    lines: List[str] = []
    lines.append(f'<a id="{github_anchor(full_name)}"></a>\n\n')
    lines.append(f"### 📌 [{full_name}]({url})\n\n")

    stats_label = "📅 Updated" if is_en else "📅 更新"
    lines.append(f"**⭐ Stars:** {stars:,} | **🍴 Forks:** {forks:,} | **{stats_label}:** {updated_at}\n\n")

    has_content = False

    brief_text = normalize_placeholder(brief, language)
    if brief_text:
        lead_label = "Brief Introduction" if is_en else "简要介绍"
        lines.append(f"**{lead_label}:** {brief_text}\n\n")
        has_content = True

    detail_pairs = [
        ("Innovations" if is_en else "创新点", normalize_placeholder(innovations, language)),
        ("Basic Usage" if is_en else "基本用法", normalize_placeholder(basic_usage, language)),
        ("Summary" if is_en else "总结", normalize_placeholder(summary_text, language)),
    ]
    bullets = [f"- **{label}:** {value}" for label, value in detail_pairs if value]
    if bullets:
        lines.extend(f"{b}\n" for b in bullets)
        lines.append("\n")
        has_content = True

    if not has_content:
        lines.append("*No AI summary available*\n\n" if is_en else "*暂无AI总结*\n\n")

    lines.append("---\n\n")
    return lines


def _select_repos_for_update(
    classified: Dict[str, List[Dict[str, Any]]],
    summary_store: Dict[str, Dict[str, Any]],
    old_summaries: Dict[str, str],
    mode: str,
    is_valid_func: callable,
) -> Dict[str, List[Dict[str, Any]]]:
    if mode != "missing_only":
        return classified
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for lang, repos in classified.items():
        needs_update = []
        for repo in repos:
            key = str(repo.get("full_name") or repo.get("Repository Name") or "").strip()
            fallback = old_summaries.get(key, "")
            summary = (summary_store.get(key, {}) or {}).get("summary") or fallback
            if not is_valid_func(str(summary or "")):
                needs_update.append(repo)
        if needs_update:
            filtered[lang] = needs_update
    return filtered


def build_readme_header(
    language: str,
    github_username: str,
    api_name: str,
    total_repos: int,
    current_time: str,
) -> List[str]:
    current_account = (github_username or "").strip()
    if current_account:
        current_account_html = f'<a href="https://github.com/{current_account}">{current_account}</a>'
    else:
        current_account_html = "未知" if language != "en" else "Unknown"

    readme_links = (
        '<a href="README.md">README（内容分类）</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a> | '
        '<a href="README_lang.md">README classified by language</a>'
    ) if language != "en" else (
        '<a href="README.md">README (content classified)</a> | '
        '<a href="README_lang.md">README classified by language</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a>'
    )

    guide_links = (
        '<a href="GUIDE_zh.md">中文教程</a> | <a href="GUIDE_en.md">English GUIDE</a>'
    ) if language != "en" else (
        '<a href="GUIDE_en.md">English GUIDE</a> | <a href="GUIDE_zh.md">中文教程</a>'
    )

    lines: List[str] = []

    if language == "en":
        lines.append(
            "<div align=\"center\">\n\n"
            "<h1>My GitHub Star Project AI Summary</h1>\n\n"
            "<p><b>Reference Repository:</b> <a href=\"https://github.com/Nanxzi/myGitStar\">Nanxzi/myGitStar</a></p>\n\n"
            f"<p>{readme_links}</p>\n"
            f"<p>{guide_links}</p>\n\n"
            "<hr/>\n\n"
            f"<p><b>Current account:</b> {current_account_html}</p>\n"
            f"<p><b>Generated on:</b> {current_time}</p>\n"
            f"<p><b>AI Model:</b> {api_name}</p>\n"
            f"<p><b>Total repositories:</b> {total_repos}</p>\n\n"
            "</div>\n\n"
        )
    else:
        lines.append(
            "<div align=\"center\">\n\n"
            "<h1>我的 GitHub Star 项目AI总结</h1>\n\n"
            "<p><b>参考仓库：</b> <a href=\"https://github.com/Nanxzi/myGitStar\">Nanxzi/myGitStar</a></p>\n\n"
            f"<p>{readme_links}</p>\n"
            f"<p>{guide_links}</p>\n\n"
            "<hr/>\n\n"
            f"<p><b>当前账号：</b> {current_account_html}</p>\n"
            f"<p><b>生成时间：</b> {current_time}</p>\n"
            f"<p><b>AI模型：</b> {api_name}</p>\n"
            f"<p><b>总仓库数：</b> {total_repos} 个</p>\n\n"
            "</div>\n\n"
        )

    return lines


def build_table_of_contents(
    classified: Dict[str, List[Dict[str, Any]]],
    language: str,
) -> List[str]:
    lines: List[str] = []
    lines.append("## 📖 目录\n\n" if language != "en" else "## 📖 Table of Contents\n\n")

    lang_counts: Dict[str, int] = {lang: len(repos) for lang, repos in classified.items()}
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        anchor = github_anchor(lang)
        count_str = f"（{count}个）" if language != "en" else f"(Total {count})"
        lines.append(f"- [{lang}](#{anchor}) {count_str}\n")

    lines.append("\n---\n\n")
    return lines


def build_repo_section(
    lang: str,
    repos: List[Dict[str, Any]],
    language: str,
    summary_store: Dict[str, Dict[str, Any]],
    old_summaries: Dict[str, Any],
    rate_limit_delay: float,
    printed_repos: set,
    printed_langs: set,
    processed_repos: int,
) -> Tuple[List[str], set, set, int]:
    if lang in printed_langs:
        return [], printed_repos, printed_langs, processed_repos

    printed_langs.add(lang)
    lines: List[str] = []
    icon = _get_lang_icon(lang)
    count_str = f"（共{len(repos)}个）" if language != "en" else f"(Total {len(repos)})"
    lines.append(f"## {icon} {lang}{count_str}\n\n")

    repos = sorted(repos, key=lambda r: -(r.get("stargazers_count") or 0))
    for repo in repos:
        if repo["full_name"] in printed_repos:
            continue
        printed_repos.add(repo["full_name"])

        summary_entry = summary_store.get(repo["full_name"], {}) or old_summaries.get(repo["full_name"], {})

        if isinstance(summary_entry, dict):
            brief = _clean_prompt_leak(summary_entry.get("Brief Introduction") or "")
            # Truncate brief to first paragraph to avoid Markdown layout issues
            if brief and ("\n" in brief or "\r" in brief):
                brief = brief.split("\n\n")[0].split("\n")[0].strip()
            innovations = _clean_prompt_leak(summary_entry.get("Innovations") or "")
            basic = _clean_prompt_leak(summary_entry.get("Basic Usage") or "")
            summary_text = _clean_prompt_leak(summary_entry.get("Summary") or "")
        elif isinstance(summary_entry, str):
            brief = _clean_prompt_leak(summary_entry)
            innovations = basic = summary_text = ""
        else:
            brief = innovations = basic = summary_text = ""

        lines.extend(
            build_repo_entry_lines(
                repo["full_name"],
                repo.get("html_url") or "",
                repo.get("stargazers_count") or 0,
                repo.get("forks_count") or 0,
                _format_updated_at(repo.get("updated_at") or ""),
                brief,
                innovations,
                basic,
                summary_text,
                language,
            )
        )
        processed_repos += 1
        time.sleep(rate_limit_delay)

    return lines, printed_repos, printed_langs, processed_repos


# === Content-based README generation ===

# Category icons for content classification
CATEGORY_ICONS = {
    "AI Agents": "🤖",
    "Autonomous Research": "🔬",
    "Productivity Tools": "⚡",
    "Creative Writing": "✍️",
    "Web Automation": "🌐",
    "Skill Development": "🧩",
    "AI Frameworks": "🧠",
    "Data Engineering": "📊",
    "DevOps & Infrastructure": "",
    "Education & Learning": "📚",
    "Other": "",
}


def _get_category_icon(category_name: str) -> str:
    """Get icon for a content category."""
    return CATEGORY_ICONS.get(category_name, "📦")


def classify_by_content(
    repos: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify repos by content category using repo_categories.json data.

    Args:
        repos: list of repo dicts (must have 'full_name')
        categories: taxonomy categories from repo_categories.json
        assignments: assignments from repo_categories.json (id + category_id)
    Returns:
        category_name -> [repo, ...]
    """
    # Build id -> category_name mapping
    cat_map: Dict[str, str] = {}
    for c in categories:
        cat_map[c["id"]] = c["name"]

    # Build full_name -> category_id mapping
    name_to_cat: Dict[str, str] = {}
    for a in assignments:
        full_name = a.get("full_name") or ""
        cat_id = a.get("category_id") or ""
        if full_name and cat_id:
            name_to_cat[full_name] = cat_id

    classified: Dict[str, List[Dict[str, Any]]] = {}
    for repo in repos:
        full_name = repo.get("full_name") or ""
        cat_id = name_to_cat.get(full_name, "")
        cat_name = cat_map.get(cat_id, "Other") if cat_id else "Other"
        classified.setdefault(cat_name, []).append(repo)

    return classified


def build_content_table_of_contents(
    classified: Dict[str, List[Dict[str, Any]]],
    language: str,
    sort_by_count: bool = True,
    descriptions: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build table of contents for content-classified README.

    Lists categories only (name, count, one-line description). Per-repo links
    are deliberately absent: with 200+ repos they turned the TOC into a
    multi-screen wall of links that buried the actual content.
    """
    lines: List[str] = []
    title = "## 📖 目录\n\n" if language != "en" else "## 📖 Table of Contents\n\n"
    lines.append(title)

    items = list(classified.items())
    if sort_by_count:
        # Sort by count descending, but put "Other" last
        other = [(k, v) for k, v in items if k == "Other"]
        non_other = [(k, v) for k, v in items if k != "Other"]
        non_other.sort(key=lambda x: -len(x[1]))
        items = non_other + other

    for cat_name, repos in items:
        anchor = github_anchor(cat_name)
        icon = _get_category_icon(cat_name)
        count = len(repos)
        count_str = f"（{count}个）" if language != "en" else f"({count})"
        desc = " ".join(str((descriptions or {}).get(cat_name) or "").split())
        if desc:
            lines.append(f"- [{icon} {cat_name}](#{anchor}) {count_str} — {desc}\n")
        else:
            lines.append(f"- [{icon} {cat_name}](#{anchor}) {count_str}\n")

    lines.append("\n---\n\n")
    return lines


def build_content_repo_section(
    category_name: str,
    repos: List[Dict[str, Any]],
    language: str,
    summary_store: Dict[str, Dict[str, Any]],
    old_summaries: Dict[str, Any],
    rate_limit_delay: float,
    printed_repos: set,
    printed_categories: set,
    processed_repos: int,
    description: str = "",
) -> Tuple[List[str], set, set, int]:
    """Build repo section for a content category (similar to build_repo_section but for content)."""
    if category_name in printed_categories:
        return [], printed_repos, printed_categories, processed_repos

    printed_categories.add(category_name)
    lines: List[str] = []
    icon = _get_category_icon(category_name)
    count_str = f"（共{len(repos)}个）" if language != "en" else f"({len(repos)})"
    lines.append(f"## {icon} {category_name} {count_str}\n\n")
    if description and description.strip():
        lines.append(f"> {description.strip()}\n\n")

    repos = sorted(repos, key=lambda r: -(r.get("stargazers_count") or 0))
    for repo in repos:
        if repo["full_name"] in printed_repos:
            continue
        printed_repos.add(repo["full_name"])

        summary_entry = summary_store.get(repo["full_name"], {}) or old_summaries.get(repo["full_name"], {})

        if isinstance(summary_entry, dict):
            brief = _clean_prompt_leak(summary_entry.get("Brief Introduction") or "")
            # Truncate brief to first paragraph to avoid Markdown layout issues
            if brief and ("\n" in brief or "\r" in brief):
                brief = brief.split("\n\n")[0].split("\n")[0].strip()
            innovations = _clean_prompt_leak(summary_entry.get("Innovations") or "")
            basic = _clean_prompt_leak(summary_entry.get("Basic Usage") or "")
            summary_text = _clean_prompt_leak(summary_entry.get("Summary") or "")
        elif isinstance(summary_entry, str):
            brief = _clean_prompt_leak(summary_entry)
            innovations = basic = summary_text = ""
        else:
            brief = innovations = basic = summary_text = ""

        lines.extend(
            build_repo_entry_lines(
                repo["full_name"],
                repo.get("html_url") or "",
                repo.get("stargazers_count") or 0,
                repo.get("forks_count") or 0,
                _format_updated_at(repo.get("updated_at") or ""),
                brief,
                innovations,
                basic,
                summary_text,
                language,
            )
        )
        processed_repos += 1
        time.sleep(rate_limit_delay)

    return lines, printed_repos, printed_categories, processed_repos


def _compute_repo_stats(repos: Optional[List[Dict[str, Any]]]) -> Tuple[int, str, int]:
    """Total stars plus the most common language across the given repos."""
    total_stars = 0
    lang_counts: Dict[str, int] = {}
    for r in repos or []:
        try:
            total_stars += int(r.get("stargazers_count") or 0)
        except Exception:
            pass
        lang = str(r.get("language") or "").strip()
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    if not lang_counts:
        return total_stars, "", 0
    top_lang, top_count = max(lang_counts.items(), key=lambda kv: kv[1])
    return total_stars, top_lang, top_count


def build_content_readme_footer(
    processed_repos: int,
    num_categories: int,
    current_time: str,
    api_name: str,
    api_call_counts: Tuple[int, int, int],
    language: str,
    repos: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Build footer for content-classified README."""
    lines: List[str] = []
    total_stars, top_lang, top_lang_count = _compute_repo_stats(repos)

    if language == "en":
        lines.append("\n## 📊 Statistics\n\n")
        lines.append(f"- **Total repositories:** {processed_repos}\n")
        lines.append(f"- **Content categories:** {num_categories}\n")
        lines.append(f"- **Total stars:** {total_stars:,}\n")
        if top_lang:
            lines.append(f"- **Top language:** {top_lang} ({top_lang_count})\n")
        lines.append(f"- **Generated on:** {current_time}\n")
        lines.append(f"- **AI Model:** {api_name}\n")
        lines.append("---\n\n")
        lines.append("*This document is generated by AI. For any errors, please refer to the original repository information.*\n")
    else:
        lines.append(f"\n## 📊 统计信息\n\n")
        lines.append(f"- **总仓库数：** {processed_repos} 个\n")
        lines.append(f"- **内容分类数：** {num_categories} 类\n")
        lines.append(f"- **总星标数：** {total_stars:,}\n")
        if top_lang:
            lines.append(f"- **最多语言：** {top_lang}（{top_lang_count} 个）\n")
        lines.append(f"- **生成时间：** {current_time}\n")
        lines.append(f"- **AI模型：** {api_name}\n")
        lines.append("---\n\n")
        lines.append("*本文档由AI自动生成，如有错误请以原仓库信息为准。*\n")

    return lines


def build_readme_footer(
    processed_repos: int,
    num_languages: int,
    current_time: str,
    api_name: str,
    api_call_counts: Tuple[int, int, int],
    language: str,
    repos: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    lines: List[str] = []
    total_stars, top_lang, top_lang_count = _compute_repo_stats(repos)

    if language == "en":
        lines.append("\n## 📊 Statistics\n\n")
        lines.append(f"- **Total repositories:** {processed_repos}\n")
        lines.append(f"- **Languages:** {num_languages}\n")
        lines.append(f"- **Total stars:** {total_stars:,}\n")
        if top_lang:
            lines.append(f"- **Top language:** {top_lang} ({top_lang_count})\n")
        lines.append(f"- **Generated on:** {current_time}\n")
        lines.append(f"- **AI Model:** {api_name}\n")
        lines.append("---\n\n")
        lines.append("*This document is generated by AI. For any errors, please refer to the original repository information.*\n")
    else:
        lines.append(f"\n## 📊 统计信息\n\n")
        lines.append(f"- **总仓库数：** {processed_repos} 个\n")
        lines.append(f"- **编程语言数：** {num_languages} 种\n")
        lines.append(f"- **总星标数：** {total_stars:,}\n")
        if top_lang:
            lines.append(f"- **最多语言：** {top_lang}（{top_lang_count} 个）\n")
        lines.append(f"- **生成时间：** {current_time}\n")
        lines.append(f"- **AI模型：** {api_name}\n")
        lines.append("---\n\n")
        lines.append("*本文档由AI自动生成，如有错误请以原仓库信息为准。*\n")

    return lines
