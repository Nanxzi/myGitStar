import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from scripts.classification.classification_parser import Taxonomy, _clean_inline_md, _strip_leading_symbols, _trim_repo_block_before_language_section
from scripts.summary.summarize_helpers import _clean_prompt_leak


def _slugify_heading(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "-")
    return s


def _build_category_anchors(taxonomy: Taxonomy) -> Dict[str, str]:
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


def _get_current_account(config: Dict[str, Any]) -> str:
    try:
        gh = config.get("github_username") if isinstance(config, dict) else None
        if gh and gh != "0" and gh != 0:
            return str(gh).strip()

        actor = (config.get("GITHUB_ACTOR") or config.get("GITHUB_USERNAME") or "").strip()
        if actor:
            return actor
    except Exception:
        return ""

    return ""


def _model_display_name(model_choice: str) -> str:
    if model_choice == "copilot":
        return "GitHub Copilot"
    if model_choice == "openrouter":
        return "OpenRouter"
    if model_choice == "gemini":
        return "Gemini"
    return model_choice


def render_classified_readme(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    config: Dict[str, Any],
    model_choice: str,
    reference_repo: str = "WuXiangM/myGitStar",
) -> str:
    categories = {c["id"]: c for c in taxonomy.categories}
    buckets: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in categories.keys()}
    other_id = next(
        (c["id"] for c in taxonomy.categories if c["name"].lower() == "other"),
        taxonomy.categories[-1]["id"],
    )

    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid, other_id)
        if cid not in buckets:
            cid = other_id
        buckets[cid].append(r)

    sort_by_count = True
    try:
        if isinstance(config, dict) and config.get("content_sort_categories_by_count") is not None:
            sort_by_count = bool(config.get("content_sort_categories_by_count"))
    except Exception:
        sort_by_count = True

    ordered_categories = taxonomy.categories
    if sort_by_count:
        ordered_categories = sorted(
            taxonomy.categories,
            key=lambda c: (
                1 if str(c.get("name", "")).strip().lower() == "other" else 0,
                -len(buckets.get(str(c.get("id")), [])),
                str(c.get("name", "")),
            ),
        )

    anchors = _build_category_anchors(Taxonomy(categories=ordered_categories))

    generated_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_name = _model_display_name(model_choice)
    total_repos = len(repos)
    current_account = _get_current_account(config)
    if current_account:
        current_account_html = f'<a href="https://github.com/{current_account}">{current_account}</a>'
    else:
        current_account_html = "Unknown"

    lines: List[str] = []
    lines.append(
        "<div align=\"center\">\n\n"
        "<h1>My GitHub Star Project AI Summary (Classified)</h1>\n\n"
        f"<p><b>Reference Repository:</b> <a href=\"https://github.com/{reference_repo}\">{reference_repo}</a></p>\n\n"
        "<p>"
        '<a href="README.md">README (content classified)</a> | '
        '<a href="README_lang.md">README classified by language</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a>'
        "</p>\n"
        "<p>"
        '<a href="GUIDE_en.md">English GUIDE</a> | '
        '<a href="GUIDE_zh.md">中文教程</a>'
        "</p>\n\n"
        "<hr/>\n\n"
        f"<p><b>Current account:</b> {current_account_html}</p>\n"
        f"<p><b>Generated on:</b> {generated_on}</p>\n"
        f"<p><b>AI Model:</b> {model_name}</p>\n"
        f"<p><b>Total repositories:</b> {total_repos}</p>\n\n"
        "</div>\n\n"
    )
    lines.append("## 📖 Table of Contents\n\n")
    for c in ordered_categories:
        anchor = anchors.get(str(c.get("id")), _slugify_heading(c.get("name", "")))
        display_name = _strip_leading_symbols(c.get("name", "")).strip() or str(c.get("name", "")).strip()
        desc = _clean_inline_md(c.get("description", ""))
        if desc:
            lines.append(f"- [{display_name}](#{anchor}) ({len(buckets.get(c['id'], []))}) — {desc}\n")
        else:
            lines.append(f"- [{display_name}](#{anchor}) ({len(buckets.get(c['id'], []))})\n")
    lines.append("\n---\n\n")

    for c in ordered_categories:
        bucket = buckets.get(c["id"], [])
        anchor = anchors.get(str(c.get("id")), "")
        if anchor:
            lines.append(f'<a id="{anchor}"></a>\n')

        display_name = _strip_leading_symbols(c.get("name", "")).strip() or str(c.get("name", "")).strip()
        lines.append(f"## {display_name} (Total {len(bucket)})\n\n")
        if c.get("description"):
            lines.append(f"> {c['description']}\n\n")

        for r in bucket:
            block_md = r.get("block_md")
            if isinstance(block_md, str) and block_md.strip():
                cleaned = _trim_repo_block_before_language_section(block_md)
                lines.append(cleaned.strip() + "\n")
            else:
                full_name = r.get("full_name") or ""
                url = r.get("html_url") or f"https://github.com/{full_name}"
                desc = (r.get("description") or "").strip()
                lines.append(f"### 📌 [{full_name}]({url})\n\n")
                if desc:
                    lines.append(desc + "\n\n")
                lines.append("---\n\n")

    return "".join(lines).rstrip() + "\n"


def render_markdown(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    config: Dict[str, Any],
    model_choice: str,
    username: str = "unknown",
    generated_at: str = None,
    language: str = "en",
) -> str:
    categories = {c["id"]: c for c in taxonomy.categories}
    buckets: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in categories.keys()}
    other_id = next(
        (c["id"] for c in taxonomy.categories if c["name"].lower() == "other"),
        taxonomy.categories[-1]["id"],
    )

    for r in repos:
        rid = r.get("id")
        cid = assignment_map.get(rid, other_id)
        if cid not in buckets:
            cid = other_id
        buckets[cid].append(r)

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    is_en = (language == "en")

    readme_links = (
        '<a href="README.md">README（内容分类）</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a> | '
        '<a href="README_lang.md">README classified by language</a>'
    ) if not is_en else (
        '<a href="README.md">README (content classified)</a> | '
        '<a href="README_lang.md">README classified by language</a> | '
        '<a href="README_lang_zh.md">README 按语言分类</a>'
    )

    guide_links = (
        '<a href="GUIDE_zh.md">中文教程</a> | <a href="GUIDE_en.md">English GUIDE</a>'
    ) if not is_en else (
        '<a href="GUIDE_en.md">English GUIDE</a> | <a href="GUIDE_zh.md">中文教程</a>'
    )

    lines: List[str] = []
    lines.append("<div align=\"center\">\n")
    if is_en:
        lines.append("\n<h1>My GitHub Star Project AI Summary</h1>\n")
        lines.append(f"\n<p><b>Reference Repository:</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n")
        lines.append(f"\n<p>{readme_links}</p>\n")
        lines.append(f"\n<p>{guide_links}</p>\n")
        lines.append("\n<hr/>\n")
        lines.append(f"\n<p><b>Current account:</b> <a href=\"https://github.com/{username}\">{username}</a></p>\n")
        lines.append(f"\n<p><b>Generated on:</b> {generated_at}</p>\n")
        lines.append(f"\n<p><b>AI Model:</b> {model_choice}</p>\n")
        lines.append(f"\n<p><b>Total repositories:</b> {len(repos)}</p>\n")
    else:
        lines.append("\n<h1>我的 GitHub Star 项目AI总结</h1>\n")
        lines.append(f"\n<p><b>参考仓库：</b> <a href=\"https://github.com/WuXiangM/myGitStar\">WuXiangM/myGitStar</a></p>\n")
        lines.append(f"\n<p>{readme_links}</p>\n")
        lines.append(f"\n<p>{guide_links}</p>\n")
        lines.append("\n<hr/>\n")
        lines.append(f"\n<p><b>当前账号：</b> <a href=\"https://github.com/{username}\">{username}</a></p>\n")
        lines.append(f"\n<p><b>生成时间：</b> {generated_at}</p>\n")
        lines.append(f"\n<p><b>AI模型：</b> {model_choice}</p>\n")
        lines.append(f"\n<p><b>总仓库数：</b> {len(repos)} 个</p>\n")
    lines.append("\n</div>\n")
    lines.append("\n---\n")
    lines.append("## Table of Contents\n" if is_en else "## 目录\n")
    sort_by_count = True
    try:
        if isinstance(config, dict) and config.get("content_sort_categories_by_count") is not None:
            sort_by_count = bool(config.get("content_sort_categories_by_count"))
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
            brief_intro = _clean_prompt_leak(repo_summary.get("Brief Introduction", ""))
            # Truncate brief to first paragraph to avoid Markdown layout issues
            if brief_intro and ("\n" in brief_intro or "\r" in brief_intro):
                brief_intro = brief_intro.split("\n\n")[0].split("\n")[0].strip()
            innovations = _clean_prompt_leak(repo_summary.get("Innovations", ""))
            basic_usage = _clean_prompt_leak(repo_summary.get("Basic Usage", ""))
            summary_text = _clean_prompt_leak(repo_summary.get("Summary", ""))

            lines.append("1. **Repository Name:** " + repo_name + "\n")
            lines.append("2. **Brief Introduction:** " + (brief_intro or "Not specified.") + "\n")
            lines.append("3. **Innovations:** " + (innovations or "Not specified.") + "\n")
            lines.append("4. **Basic Usage:** " + (basic_usage or "Not specified.") + "\n")
            lines.append("5. **Summary:** " + (summary_text or "Not specified.") + "\n")

            lines.append("---\n\n")

    return "\n".join(lines).strip() + "\n"


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def apply_min_repos_per_category(
    taxonomy: Taxonomy,
    repos: List[Dict[str, Any]],
    assignment_map: Dict[Any, str],
    min_repos_per_category: int,
    min_categories: int,
) -> Tuple[Taxonomy, Dict[Any, str]]:
    if min_repos_per_category <= 0:
        return taxonomy, assignment_map

    if not taxonomy.categories:
        return taxonomy, assignment_map

    other_old = next(
        (c["id"] for c in taxonomy.categories if str(c.get("name", "")).strip().lower() == "other"),
        taxonomy.categories[-1]["id"],
    )

    counts: Dict[str, int] = {str(c["id"]): 0 for c in taxonomy.categories}
    for r in repos:
        rid = r.get("id")
        cid = str(assignment_map.get(rid, other_old))
        if cid not in counts:
            cid = other_old
        counts[cid] = counts.get(cid, 0) + 1

    candidates: List[Tuple[str, int]] = []
    for c in taxonomy.categories:
        if str(c.get("name", "")).strip().lower() == "other":
            continue
        cid = str(c["id"])
        cnt = int(counts.get(cid, 0) or 0)
        if cnt < min_repos_per_category:
            candidates.append((cid, cnt))
    candidates.sort(key=lambda x: (x[1], x[0]))

    drop_ids: set[str] = set()
    remaining_count = len(taxonomy.categories)
    min_keep = max(1, int(min_categories))
    for cid, _cnt in candidates:
        if remaining_count - 1 < min_keep:
            break
        drop_ids.add(cid)
        remaining_count -= 1

    if not drop_ids:
        return taxonomy, assignment_map

    remaining = [
        c for c in taxonomy.categories if str(c["id"]) not in drop_ids and str(c.get("name", "")).strip().lower() != "other"
    ]
    other_cat = next(
        (c for c in taxonomy.categories if str(c.get("name", "")).strip().lower() == "other"),
        taxonomy.categories[-1],
    )

    new_categories: List[Dict[str, Any]] = []
    id_map: Dict[str, str] = {}
    for idx, c in enumerate(remaining, start=1):
        new_id = f"C{idx}"
        id_map[str(c["id"])] = new_id
        new_categories.append({"id": new_id, "name": c.get("name", ""), "description": c.get("description", "")})

    other_new_id = f"C{len(new_categories) + 1}"
    id_map[str(other_cat["id"])] = other_new_id
    new_categories.append(
        {
            "id": other_new_id,
            "name": "Other",
            "description": other_cat.get("description", "") or "Everything that does not fit other categories.",
        }
    )

    new_assignment: Dict[Any, str] = {}
    for r in repos:
        rid = r.get("id")
        old_cid = str(assignment_map.get(rid, other_old))
        if old_cid in drop_ids:
            new_assignment[rid] = other_new_id
            continue
        new_assignment[rid] = id_map.get(old_cid, other_new_id)

    return Taxonomy(categories=new_categories), new_assignment
