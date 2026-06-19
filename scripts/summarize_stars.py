import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
from typing import Any, Dict, List

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.core.config import (
    load_config,
    get_int_config,
    get_float_config,
    env_truthy,
    resolve_update_mode,
    normalize_update_mode,
)
from scripts.core.secrets import load_api_keys
from scripts.core.throttle import SimpleThrottle
from scripts.core.json_store import (
    get_summary_json_path,
    build_summary_index,
    save_json_atomic,
    merge_summary_store,
    load_summary_store,
    compute_description_hash,
    make_metadata,
    is_entry_fresh,
)
from scripts.github import get_starred_repos
from scripts.output import (
    classify_by_language,
    build_readme_header,
    build_table_of_contents,
    build_repo_section,
    build_readme_footer,
)
from scripts.summary import (
    build_repo_entry,
    is_valid_summary,
    select_repos_for_update,
    summarize_batch,
    summarize_batch_combined,
    get_summarize_func,
)


config = load_config()

DEBUG_API = env_truthy("DEBUG_API") or bool(config.get("test_first_repo", False))

GITHUB_TOKEN, OPENROUTER_API_KEY, GEMINI_API_KEY, MODELSCOPE_API_KEY = load_api_keys(config)

update_mode = resolve_update_mode(config)

print(f"DEBUG: config loaded: github_username={repr(config.get('github_username'))}, model_choice={repr(config.get('model_choice'))}")
print(f"DEBUG: config keys: {list(config.keys()) if isinstance(config, dict) else 'not a dict'}")

github_username = config.get("github_username")
model_choice = config.get("model_choice", "copilot")

default_copilot_model = config.get("default_copilot_model")
default_openrouter_model = config.get("default_openrouter_model")
default_gemini_model = config.get("default_gemini_model", "gemini-pro")
default_modelscope_model = config.get("default_modelscope_model", "deepseek-ai/DeepSeek-V3.2")

max_workers = get_int_config(config, "max_workers", 5)
batch_size = get_int_config(config, "batch_size", 1)
batch_mode = config.get("batch_mode", "concurrent").lower()
request_timeout = get_float_config(config, "request_timeout", 10.0)
rate_limit_delay = get_float_config(config, "rate_limit_delay", 1.0)
request_retry_delay = get_int_config(config, "request_retry_delay", 5)
retry_attempts = get_int_config(config, "retry_attempts", 3)
readme_sum_path = config.get("readme_sum_path")

if github_username == "0" or github_username == 0:
    GITHUB_USERNAME = os.environ.get("GITHUB_ACTOR") or os.environ.get("GITHUB_USERNAME")
    if not GITHUB_USERNAME:
        print("未检测到 workflow 账号环境变量 GITHUB_ACTOR/GITHUB_USERNAME，请检查 workflow 配置！")
    else:
        print(f"DEBUG: github_username from config={repr(github_username)}, GITHUB_USERNAME from env={repr(GITHUB_USERNAME)}")
else:
    GITHUB_USERNAME = github_username
    print(f"DEBUG: github_username from config={repr(github_username)}, using config value GITHUB_USERNAME={repr(GITHUB_USERNAME)}")

MAX_REPOS: int = None
max_repos_env = os.environ.get("MAX_REPOS")
if max_repos_env:
    try:
        mr = int(max_repos_env)
        if mr > 0:
            MAX_REPOS = mr
    except Exception:
        MAX_REPOS = None

if MAX_REPOS is None:
    try:
        cfg_mr = config.get("max_repos") if isinstance(config, dict) else None
        if cfg_mr is not None:
            mr = int(cfg_mr)
            if mr > 0:
                MAX_REPOS = mr
    except Exception:
        pass

GLOBAL_QPS = get_float_config(config, "global_qps", 0.5)
THROTTLE = SimpleThrottle(GLOBAL_QPS)

copilot_api_call_count = 0
openrouter_api_call_count = 0
gemini_api_call_count = 0
modelscope_api_call_count = 0


def _repo_key(repo: Dict) -> str:
    return str(repo.get("full_name") or repo.get("Repository Name") or "").strip()


def _api_call_counter():
    global copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count, modelscope_api_call_count
    if model_choice == "copilot":
        copilot_api_call_count += 1
        remaining = 150 - copilot_api_call_count
        print(f"[Copilot API调用] 第 {copilot_api_call_count} 次调用，剩余可用: {remaining}")
    elif model_choice == "openrouter":
        openrouter_api_call_count += 1
        print(f"[OpenRouter API调用] 第 {openrouter_api_call_count} 次调用")
    elif model_choice == "gemini":
        gemini_api_call_count += 1
        print(f"[Gemini API调用] 第 {gemini_api_call_count} 次调用")
    elif model_choice == "modelscope":
        modelscope_api_call_count += 1
        print(f"[ModelScope API调用] 第 {modelscope_api_call_count} 次调用")


README_SUM_PATH = readme_sum_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "README-sum.md")
LANGUAGE = config.get("language", "zh")

LOG_FILE = config.get("log_file", os.path.join(os.path.dirname(__file__), "summarize_stars.log"))
LOG_MAX_BYTES = get_int_config(config, "log_max_bytes", 5 * 1024 * 1024)
LOG_BACKUP_COUNT = get_int_config(config, "log_backup_count", 3)

logger = logging.getLogger("summarize_stars")
logger.setLevel(logging.DEBUG if DEBUG_API else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

orig_stdout = sys.stdout
orig_stderr = sys.stderr


class TeeStream:
    def __init__(self, orig, lg, level):
        self.orig = orig
        self.lg = lg
        self.level = level

    def write(self, msg):
        try:
            self.orig.write(msg)
        except Exception:
            pass
        if msg and msg.strip():
            try:
                self.lg.log(self.level, msg.rstrip())
            except Exception:
                pass

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


sys.stdout = TeeStream(orig_stdout, logger, logging.INFO)
sys.stderr = TeeStream(orig_stderr, logger, logging.ERROR)

if OPENROUTER_API_KEY:
    print(f"OpenRouter API Key 前缀: {OPENROUTER_API_KEY[:6]}...")
if GITHUB_TOKEN:
    print(f"GitHub Token 前缀: {GITHUB_TOKEN[:6]}...")
if GEMINI_API_KEY:
    try:
        print(f"Gemini API Key 前缀: {GEMINI_API_KEY[:4]}...")
    except Exception:
        print("Gemini API Key 前缀: (已设置)")
if MODELSCOPE_API_KEY:
    print(f"ModelScope API Key 前缀: {MODELSCOPE_API_KEY[:6]}...")


def main():
    if model_choice:
        api_choice = model_choice.lower()
    else:
        api_choice = "copilot" if os.environ.get("USE_COPILOT_API", "true").lower() == "true" else "openrouter"

    if api_choice == "gemini":
        api_name = "Gemini"
    elif api_choice == "openrouter":
        api_name = "OpenRouter (DeepSeek)"
    elif api_choice == "modelscope":
        api_name = "ModelScope (DeepSeek-V3.2)"
    else:
        api_name = "GitHub Copilot"

    print(f"开始使用 {api_name} 生成 GitHub Star 项目总结...")
    print(f"[mode] update_mode={update_mode} (missing_only=仅补缺失/新增；all=全量重汇总)")

    try:
        starred = get_starred_repos(GITHUB_TOKEN, GITHUB_USERNAME, THROTTLE, request_timeout, MAX_REPOS)

        try:
            test_first_repo = bool(config.get("test_first_repo", False))
        except Exception:
            test_first_repo = False
        if test_first_repo and isinstance(starred, list) and len(starred) > 0:
            print("[TEST MODE] test_first_repo 已启用：仅处理第一个仓库进行调试")
            starred = [starred[0]]

        if MAX_REPOS and isinstance(starred, list):
            try:
                limit = int(MAX_REPOS)
                if limit > 0 and len(starred) > limit:
                    print(f"[LIMIT] 因环境变量 MAX_REPOS={limit}，仅处理前 {limit} 个仓库以避免超时")
                    starred = starred[:limit]
            except Exception:
                pass

        classified = classify_by_language(starred)

        json_path = get_summary_json_path(LANGUAGE)
        summary_store = load_summary_store(json_path)
        old_summaries = build_summary_index(summary_store)

        # Build a description_lookup so we can do hash-based incremental
        # selection (the legacy code only checked the *rendered* summary text).
        description_lookup: Dict[str, str] = {}
        for repo in starred:
            key = str(repo.get("full_name") or "").strip()
            if key:
                description_lookup[key] = str(repo.get("description") or "")

        from scripts.core.summary_reader import load_old_summaries
        if not old_summaries:
            old_summaries = load_old_summaries(json_path, README_SUM_PATH, LANGUAGE)
        if not summary_store and old_summaries:
            for full_name, summary in old_summaries.items():
                summary_store[full_name] = {"full_name": full_name, "summary": summary}

        # Diagnostic: how many existing entries are already "fresh" (won't be
        # re-summarised this run) and how many are unknown/stale.
        fresh_count = 0
        stale_unknown = []
        for repo in starred:
            key = str(repo.get("full_name") or "").strip()
            if not key:
                continue
            entry = summary_store.get(key) or {}
            if not isinstance(entry, dict):
                entry = {}
            if is_entry_fresh(entry, key, description_lookup.get(key, ""), refresh_after_days=0):
                fresh_count += 1
            else:
                # Only flag truly unknown (no Summary at all) for the diagnostic
                if not (entry.get("Summary") or "").strip():
                    stale_unknown.append(key)
        print(f"[DIAG] {fresh_count}/{len(starred)} repos already fresh; "
              f"{len(stale_unknown)} still unknown: {stale_unknown[:5]}"
              f"{'...' if len(stale_unknown) > 5 else ''}")

        repos_to_update = select_repos_for_update(
            classified,
            old_summaries,
            update_mode,
            LANGUAGE,
            description_lookup=description_lookup,
        )

        # API call budget: limit LLM calls per run. Default 0 (unlimited) so
        # local runs are not constrained; the workflow sets MAX_API_CALLS env.
        max_api_calls_env = os.environ.get("MAX_API_CALLS", "")
        max_api_calls_cfg = config.get("max_api_calls_per_run", 0)
        try:
            max_api_calls = int(max_api_calls_env) if max_api_calls_env else int(max_api_calls_cfg or 0)
        except Exception:
            max_api_calls = 0
        _api_calls_used = {"n": 0}
        if max_api_calls and max_api_calls > 0:
            def _budget_tracker() -> bool:
                return _api_calls_used["n"] < max_api_calls
            print(f"[BUDGET] max_api_calls={max_api_calls}")
        else:
            def _budget_tracker() -> bool:
                return True

        # Wrap summarize_func so we can also count API calls in the budget
        # (the per-call counter in _api_call_counter stays unchanged).
        _orig_summarize_func = None
        def _budgeted_summarize_func(repo_dict):
            if not _budget_tracker():
                # Returning None tells the caller to preserve old summary.
                print(f"[BUDGET] skipping call, would exceed {max_api_calls} limit")
                return None
            result = _orig_summarize_func(repo_dict)
            _api_calls_used["n"] += 1
            return result

        classified_to_process: Dict[str, List[Dict]] = {}
        for lang, repos in classified.items():
            try:
                sorted_repos = sorted(repos, key=lambda r: is_valid_summary(old_summaries.get(r.get("full_name", ""), ""), LANGUAGE))
            except Exception:
                sorted_repos = repos
            if sorted_repos:
                classified_to_process[lang] = sorted_repos

        current_time = time.strftime("%Y-%m-%d", time.localtime())

        lines: List[str] = []
        lines.extend(build_readme_header(LANGUAGE, GITHUB_USERNAME, api_name, len(starred), current_time))
        lines.extend(build_table_of_contents(classified_to_process, LANGUAGE))

        printed_repos: set = set()
        printed_langs: set = set()
        total_repos = sum(len(repos) for repos in classified_to_process.values())
        processed_repos = 0
        repo_summary_map: Dict[str, Dict] = {}

        summarize_func = get_summarize_func(
            model_choice=api_choice,
            github_token=GITHUB_TOKEN,
            openrouter_api_key=OPENROUTER_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            modelscope_api_key=MODELSCOPE_API_KEY,
            default_copilot_model=default_copilot_model,
            default_openrouter_model=default_openrouter_model,
            default_gemini_model=default_gemini_model,
            default_modelscope_model=default_modelscope_model,
            language=LANGUAGE,
            config=config,
            throttle=THROTTLE,
            request_timeout=request_timeout,
            request_retry_delay=float(request_retry_delay),
            retry_attempts=retry_attempts,
            api_call_counter=_api_call_counter,
        )
        _orig_summarize_func = summarize_func

        # Pick the model name to stamp into metadata.
        if api_choice == "copilot":
            _model_name_for_meta = default_copilot_model or "copilot"
        elif api_choice == "openrouter":
            _model_name_for_meta = default_openrouter_model or "openrouter"
        elif api_choice == "gemini":
            _model_name_for_meta = default_gemini_model or "gemini"
        elif api_choice == "modelscope":
            _model_name_for_meta = default_modelscope_model or "modelscope"
        else:
            _model_name_for_meta = api_choice

        all_repos_to_process: List[Dict] = []
        for lang, repos in classified_to_process.items():
            # `repos_to_update` already honours update_mode + description_hash
            # (see select_repos_for_update), so we just consume its result.
            repos_to_call = repos_to_update.get(lang, [])
            all_repos_to_process.extend(repos_to_call)

        printed_repos: set = set()
        printed_langs: set = set()
        total_repos = len(all_repos_to_process)
        processed_repos = 0
        repo_summary_map: Dict[str, Dict] = {}

        for i in range(0, len(all_repos_to_process), batch_size):
            this_batch = all_repos_to_process[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"处理批次 {batch_num}，包含 {len(this_batch)} 个仓库...")

            if batch_mode == "combined" and batch_size > 1:
                print(f"[DEBUG] Calling summarize_batch_combined for {len(this_batch)} repos, batch_size={batch_size}")
                summaries = summarize_batch_combined(
                    this_batch,
                    old_summaries,
                    _budgeted_summarize_func,
                    update_mode,
                    LANGUAGE,
                    batch_size,
                    batch_num,
                    api_budget_tracker=_budget_tracker,
                    description_lookup=description_lookup,
                    model_name=_model_name_for_meta,
                )
            else:
                summaries = summarize_batch(
                    this_batch,
                    old_summaries,
                    _budgeted_summarize_func,
                    update_mode,
                    LANGUAGE,
                    max_workers,
                )

            for repo, summary in zip(this_batch, summaries):
                key = _repo_key(repo)
                entry = build_repo_entry(repo, summary)
                if key:
                    repo_summary_map[key] = entry

            summary_store = merge_summary_store(summary_store, repo_summary_map)
            save_json_atomic(summary_store, json_path)

        for lang in sorted(classified_to_process.keys(), key=lambda x: -len(classified_to_process[x])):
            if lang in printed_langs:
                continue
            repos = classified_to_process[lang]

            section_lines, printed_repos, printed_langs, processed_repos = build_repo_section(
                lang,
                repos,
                LANGUAGE,
                summary_store,
                old_summaries,
                rate_limit_delay,
                printed_repos,
                printed_langs,
                processed_repos,
            )
            lines.extend(section_lines)

        # Final diagnostic: count budget spent, still-unknown repos, and
        # last-errors so the workflow log makes it obvious what to do next.
        if max_api_calls and max_api_calls > 0:
            print(f"[BUDGET] used {_api_calls_used['n']}/{max_api_calls} LLM calls this run")
        still_unknown = [
            k for k, v in (summary_store or {}).items()
            if not (isinstance(v, dict) and (v.get("Summary") or "").strip())
        ]
        if still_unknown:
            reasons: Dict[str, str] = {}
            for k in still_unknown:
                entry = summary_store.get(k) or {}
                if not isinstance(entry, dict):
                    reasons[k] = "no_entry"
                else:
                    reasons[k] = entry.get("__last_error__") or "no_summary_field"
            print(f"[DIAG] {len(still_unknown)} repos still unknown after this run:")
            for k in still_unknown[:20]:
                print(f"  - {k}: {reasons.get(k, '?')}")

        lines.extend(
            build_readme_footer(
                processed_repos,
                len(classified_to_process),
                current_time,
                api_name,
                (copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count),
                LANGUAGE,
            )
        )

        with open(README_SUM_PATH, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        print(f"\n✅ {README_SUM_PATH} 已生成，共处理了 {processed_repos} 个仓库。")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate language-classified README summaries.")
    parser.add_argument("--language", type=str, default=None, help="Override language: en or zh.")
    parser.add_argument("--out", type=str, default=None, help="Override output markdown path.")
    parser.add_argument(
        "--update-mode",
        type=str,
        default=None,
        help="Override update mode: missing_only or all (also supports env MYGITSTAR_UPDATE_MODE).",
    )
    parser.add_argument("--copilot-count", action="store_true", help="Print Copilot API call count (for this run) and exit.")
    args = parser.parse_args()

    if args.copilot_count or (len(sys.argv) > 1 and sys.argv[1] == "--copilot-count"):
        print(copilot_api_call_count)
        raise SystemExit(0)

    if args.language:
        lang = str(args.language).strip().lower()
        if lang in {"cn", "zh-cn", "zh_cn", "zh"}:
            LANGUAGE = "zh"
        elif lang in {"en", "eng", "english"}:
            LANGUAGE = "en"
        else:
            print(f"Unsupported --language: {args.language}")
            raise SystemExit(2)

    if args.out:
        out_path = str(args.out).strip()
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), out_path)
        README_SUM_PATH = out_path

    if args.update_mode is not None:
        update_mode = normalize_update_mode(args.update_mode)

    main()