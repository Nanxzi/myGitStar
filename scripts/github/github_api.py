import base64
import time
from typing import Any, Dict, List, Optional

import requests

from scripts.core.throttle import SimpleThrottle


def get_starred_repos(
    github_token: str,
    github_username: str,
    throttle: Optional[SimpleThrottle] = None,
    timeout: float = 30.0,
    max_repos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not github_token:
        raise ValueError("缺少 GITHUB_TOKEN 环境变量")

    repos: List[Dict[str, Any]] = []
    page = 1
    per_page = 100
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"https://api.github.com/users/{github_username}/starred?per_page={per_page}&page={page}"
        try:
            if throttle:
                try:
                    throttle.wait()
                except Exception:
                    pass
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            repos.extend(data)
            print(f"已获取 {len(repos)} 个仓库... (第 {page} 页)")
            page += 1
            time.sleep(1)

            if max_repos and max_repos > 0 and len(repos) >= max_repos:
                repos = repos[:max_repos]
                break

        except requests.RequestException as e:
            print(f"获取星标仓库失败: {e}")
            break

    print(f"总共获取到 {len(repos)} 个星标仓库")
    return repos


def fetch_repo_readme(
    github_token: str,
    full_name: str,
    timeout: float = 15.0,
    max_chars: int = 3000,
) -> str:
    """Fetch and decode a repo's README, truncated to max_chars.

    Uses GET /repos/{owner}/{repo}/readme which returns base64-encoded content.
    Returns empty string on any failure (non-critical data).
    """
    if not github_token:
        return ""
    try:
        url = f"https://api.github.com/repos/{full_name}/readme"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        content_b64 = data.get("content", "")
        if not content_b64:
            return ""
        decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        # Strip markdown code blocks and image links to save tokens
        decoded = _strip_markdown_noise(decoded)
        return decoded[:max_chars]
    except Exception:
        return ""


def _strip_markdown_noise(text: str) -> str:
    """Remove heavy markdown elements that waste LLM tokens."""
    import re
    # Remove image tags
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
