import json
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
    count_invalid_fields,
    entry_attempts,
    is_retry_due,
    is_source_absent,
    load_queue_state,
    save_queue_state,
)
from scripts.github import get_starred_repos, fetch_repo_readme
from scripts.output import (
    classify_by_language,
    build_readme_header,
    build_table_of_contents,
    build_repo_section,
    build_readme_footer,
    classify_by_content,
    build_content_table_of_contents,
    build_content_repo_section,
    build_content_readme_footer,
)
from scripts.summary import (
    build_repo_entry,
    is_valid_summary,
    select_repos_for_update,
    summarize_batch,
    summarize_batch_combined,
    get_summarize_func,
)
from scripts.ai.llm_caller import RateLimitAbort, set_max_consecutive_429, _reset_consecutive_429, _note_429_and_maybe_abort
from scripts.core import daily_counter


config = load_config()

# Apply max_consecutive_429 from config to llm_caller module
set_max_consecutive_429(get_int_config(config, "max_consecutive_429", 3))

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


def _priority_key(repo: Dict, summary_store: Dict[str, Dict]):
    """Sort key for the summarization work queue.

    Lower tuple = processed earlier:
      1. repos whose failure backoff has expired come first;
      2. among those, repos with MORE invalid content fields first
         (core requirement: multi-empty-field repos get the budget first);
      3. fewer recorded attempts first (fairness across runs);
      4. full_name as a deterministic tiebreaker.
    """
    key = _repo_key(repo)
    entry = summary_store.get(key) or {}
    if not isinstance(entry, dict):
        entry = {}
    return (
        0 if is_retry_due(entry) else 1,
        -count_invalid_fields(entry),
        entry_attempts(entry),
        key,
    )


def _cohort_tier(repo: Dict, summary_store: Dict[str, Dict]):
    """Priority tier used to group repos for round-robin rotation."""
    key = _repo_key(repo)
    entry = summary_store.get(key) or {}
    if not isinstance(entry, dict):
        entry = {}
    return (-count_invalid_fields(entry), entry_attempts(entry))


def _rotate_cohorts(work_list: List[Dict], summary_store: Dict[str, Dict], cursor: int) -> List[Dict]:
    """Round-robin within each equal-priority cohort (P1-4).

    Repos in the same tier (same invalid-field count and attempt count) are
    rotated by `cursor % cohort_size` so consecutive runs spend the budget on
    different slices of the cohort instead of always the alphabetical head.
    """
    if not work_list:
        return work_list
    out: List[Dict] = []
    i = 0
    while i < len(work_list):
        tier = _cohort_tier(work_list[i], summary_store)
        j = i
        while j < len(work_list) and _cohort_tier(work_list[j], summary_store) == tier:
            j += 1
        cohort = work_list[i:j]
        if len(cohort) > 1 and cursor > 0:
            k = cursor % len(cohort)
            out.extend(cohort[k:] + cohort[:k])
        else:
            out.extend(cohort)
        i = j
    return out


def _api_call_counter(primary: bool = True):
    """Count a summarize call against the primary provider's quotas.

    `primary=False` marks a call served by a fallback provider: it does not
    consume the primary's per-run / RPD quota, so a run whose primary is
    rate-limited can keep making progress via fallbacks without tripping
    the abort below on every call.
    """
    global copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count, modelscope_api_call_count
    if not primary:
        return
    if model_choice == "copilot":
        copilot_api_call_count += 1
        # 从 config 读取 max_api_calls_per_run（0 = 不限制）
        max_calls = config.get("max_api_calls_per_run", 0)
        remaining = max_calls - copilot_api_call_count if max_calls > 0 else None
        if max_calls > 0:
            print(f"[Copilot API调用] 第 {copilot_api_call_count} 次调用，剩余可用: {remaining}")
            # Stop when Copilot quota is exhausted
            if remaining is not None and remaining <= 0:
                raise RateLimitAbort(
                    f"Copilot API quota exhausted: {copilot_api_call_count} calls made, remaining={remaining}. "
                    "Stopping to avoid further overuse. Results will be saved and can be resumed next run."
                )
        else:
            print(f"[Copilot API调用] 第 {copilot_api_call_count} 次调用（无上限）")
    elif model_choice == "openrouter":
        openrouter_api_call_count += 1
        # Check RPD limit BEFORE the call (increment is done in openrouter_summarize via check_and_reserve)
        rpd_limit = get_int_config(config, "openrouter_rpd", 50)
        # Reserve some RPD for the classify stage so summarize can never
        # starve it on a shared free-tier quota (P1-5).
        rpd_reserve = get_int_config(config, "classify_rpd_reserve", 5)
        allowed, current = daily_counter.check_limit(rpd_limit, reserve=rpd_reserve)
        if not allowed:
            raise RateLimitAbort(
                f"OpenRouter RPD limit reached: {current}/{rpd_limit} calls today "
                f"(with {rpd_reserve} reserved for classify). "
                "Stopping to avoid further rate limit errors. Results will be saved."
            )
        print(f"[OpenRouter API调用] 第 {openrouter_api_call_count} 次调用，今日已用: {current}/{rpd_limit}")
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


def _run_plan_only() -> int:
    """Offline priority-queue preview (P2-3): no API calls, no file writes."""
    json_path = get_summary_json_path(LANGUAGE)
    store = load_summary_store(json_path)
    if not store:
        print(f"[PLAN] {json_path} is empty or missing — nothing to plan.")
        return 0

    work: List[Dict] = []
    backing_off = 0
    absent = 0
    for full_name, entry in store.items():
        if not isinstance(entry, dict):
            entry = {}
        if is_source_absent(entry):
            absent += 1
            continue
        if not is_retry_due(entry):
            backing_off += 1
            continue
        if count_invalid_fields(entry) > 0:
            work.append({"full_name": full_name})

    store_for_sort = dict(store)
    work.sort(key=lambda r: _priority_key(r, store_for_sort))

    total = len(store)
    invalid = sum(1 for e in store.values() if isinstance(e, dict) and count_invalid_fields(e) > 0)
    empty = sum(1 for e in store.values() if not (isinstance(e, dict) and (e.get("Summary") or "").strip()))
    queue_state = load_queue_state()

    print(f"[PLAN] store={json_path}")
    print(f"[PLAN] entries={total} | invalid-field={invalid} | empty-summary={empty}")
    print(f"[PLAN] queue depth={len(work)} | backing off={backing_off} | source_absent={absent} | cursor={queue_state.get('cursor', 0)}")

    bs = get_int_config(config, "batch_size", 5)
    budget = get_int_config(config, "max_api_calls_per_run", 0)
    if work and bs > 0:
        capacity = (budget if budget > 0 else len(work)) * bs
        print(f"[PLAN] suggested coverage this run: ~{min(capacity, len(work))}/{len(work)} repos "
              f"(budget={budget or 'unlimited'} calls x batch_size={bs}, combined mode)")
    if work:
        print("[PLAN] top 20 by priority (invalid_fields, attempts, name):")
        for r in work[:20]:
            key = r["full_name"]
            entry = store.get(key) or {}
            print(f"  - {key}: invalid={count_invalid_fields(entry)}, attempts={entry_attempts(entry)}")
    return 0


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

        # Convergence baseline (P2-2): snapshot before the run so we can
        # report how many invalid-field repos this run actually fixed.
        _invalid_before = sum(
            1 for e in (summary_store or {}).values()
            if isinstance(e, dict) and count_invalid_fields(e) > 0
        )
        _empty_summary_before = sum(
            1 for e in (summary_store or {}).values()
            if not (isinstance(e, dict) and (e.get("Summary") or "").strip())
        )
        print(f"[CONVERGENCE] before run: {_invalid_before} repos with >=1 invalid field, "
              f"{_empty_summary_before} with empty Summary")

        repos_to_update = select_repos_for_update(
            classified,
            summary_store,
            update_mode,
            LANGUAGE,
            description_lookup=description_lookup,
        )

        # Backfill __meta__ for legacy entries that carry valid content but
        # no metadata (e.g. old pure-string entries). This is OFF-LINE — no
        # LLM call — and works for both combined and concurrent modes. Once
        # stamped with the current description_hash, such entries participate
        # in hash-based incremental selection on future runs instead of being
        # (a) permanently re-summarised on force_all or (b) silently skipped
        # as "fresh" with no way to detect upstream description changes.
        _backfilled = 0
        for _key, _entry in (summary_store or {}).items():
            _desc = description_lookup.get(_key, "").strip()
            if not _desc:
                # No upstream description (repo no longer starred / no desc):
                # leave as-is, we cannot compute a trustworthy hash.
                continue
            if not isinstance(_entry, dict):
                continue
            if not (str(_entry.get("Summary") or "").strip()):
                # No content yet -> needs a real LLM call, handle elsewhere.
                continue
            _meta = _entry.get("__meta__")
            if isinstance(_meta, dict) and _meta.get("description_hash"):
                continue  # already stamped
            _entry["__meta__"] = make_metadata(
                full_name=_key,
                description=_desc,
                model=(_meta.get("summary_model") if isinstance(_meta, dict) else None) or "legacy",
                source="backfill",
                attempts=0,
            )
            _backfilled += 1
        if _backfilled:
            save_json_atomic(summary_store, json_path)
            print(f"[BACKFILL] Stamped __meta__ for {_backfilled} fresh-but-legacy entries (no API calls)")

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
        # Double ledger (P1-5):
        #   - progress budget (max_api_calls): ONLY calls that returned usable
        #     text are counted; failures burn the consecutive-failure budget
        #     instead, so a bad API day doesn't waste the whole quota;
        #   - hard quotas (Copilot per-run cap / OpenRouter RPD) still count
        #     every attempt inside _api_call_counter / daily_counter.
        _orig_summarize_func = None
        def _budgeted_summarize_func(repo_dict):
            if not _budget_tracker():
                # Returning None tells the caller to preserve old summary.
                print(f"[BUDGET] skipping call, would exceed {max_api_calls} limit")
                return None
            result = _orig_summarize_func(repo_dict)
            # Only calls that produced usable text count against the budget.
            produced_output = isinstance(result, str) and bool(result.strip())
            if not produced_output:
                if result is not None:
                    print(f"[BUDGET] call produced no usable output, not counting against budget")
                return result
            _api_calls_used["n"] += 1
            return result

        classified_to_process: Dict[str, List[Dict]] = {}
        for lang, repos in classified.items():
            try:
                # Priority ordering (P1-2): most-invalid-first, then fairness.
                # Key: (backoff due first, more invalid fields first, fewer
                # attempts first, deterministic name tiebreaker).
                sorted_repos = sorted(repos, key=lambda r: _priority_key(r, summary_store))
            except Exception:
                sorted_repos = repos
            if sorted_repos:
                classified_to_process[lang] = sorted_repos

        current_time = time.strftime("%Y-%m-%d", time.localtime())

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
            repos_to_call = repos_to_update.get(lang, [])
            all_repos_to_process.extend(repos_to_call)

        # --- Priority queue filtering (P1-3) ---
        # Terminal-state and still-backing-off repos do not consume budget.
        queue_state = load_queue_state()
        queue_cursor = int(queue_state.get("cursor") or 0)
        filtered_out_backoff = 0
        filtered_out_absent = 0
        work_queue: List[Dict] = []
        for repo in all_repos_to_process:
            key = _repo_key(repo)
            entry = summary_store.get(key) or {}
            if not isinstance(entry, dict):
                entry = {}
            if is_source_absent(entry):
                filtered_out_absent += 1
                continue
            if not is_retry_due(entry):
                filtered_out_backoff += 1
                continue
            work_queue.append(repo)
        if filtered_out_absent or filtered_out_backoff:
            print(
                f"[QUEUE] filtered {len(all_repos_to_process)} selected repos -> "
                f"{len(work_queue)} workable (backoff: {filtered_out_backoff}, source_absent: {filtered_out_absent})"
            )

        # --- Round-robin rotation within equal-priority cohorts (P1-4) ---
        if work_queue:
            work_queue = _rotate_cohorts(work_queue, summary_store, queue_cursor)

        all_repos_to_process = work_queue

        # Fetch README content for repos that need summarization (token control)
        if all_repos_to_process:
            print(f"\n[README] 抓取 {len(all_repos_to_process)} 个仓库的 README...")
            readme_fetched = 0
            readme_failed = 0
            for repo in all_repos_to_process:
                full_name = repo.get("full_name") or ""
                if not full_name:
                    continue
                content = fetch_repo_readme(GITHUB_TOKEN, full_name, timeout=15.0, max_chars=3000)
                if content:
                    repo["readme_content"] = content
                    readme_fetched += 1
                else:
                    readme_failed += 1
            print(f"[README] 成功 {readme_fetched}, 失败 {readme_failed}")

        printed_repos: set = set()
        printed_langs: set = set()
        total_repos = len(all_repos_to_process)
        processed_repos = 0
        consecutive_failed_batches = 0
        max_consecutive_failures = get_int_config(config, "max_consecutive_failures", 5)
        backoff_base_days = get_float_config(config, "retry_backoff_base_days", 1.0)
        backoff_max_days = get_float_config(config, "retry_backoff_max_days", 8.0)
        rate_limit_batch_retries = get_int_config(config, "rate_limit_batch_retries", 2)
        rate_limit_retry_wait = get_float_config(config, "rate_limit_retry_wait", 60.0)

        for i in range(0, len(all_repos_to_process), batch_size):
            this_batch = all_repos_to_process[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"处理批次 {batch_num}，包含 {len(this_batch)} 个仓库...")

            try:
                if batch_mode == "combined" and batch_size > 1:
                    print(f"[DEBUG] Calling summarize_batch_combined for {len(this_batch)} repos, batch_size={batch_size}")
                    summaries = summarize_batch_combined(
                        this_batch,
                        summary_store,
                        _budgeted_summarize_func,
                        update_mode,
                        LANGUAGE,
                        batch_size,
                        batch_num,
                        api_budget_tracker=_budget_tracker,
                        description_lookup=description_lookup,
                        model_name=_model_name_for_meta,
                        backoff_base_days=backoff_base_days,
                        backoff_max_days=backoff_max_days,
                        rate_limit_batch_retries=rate_limit_batch_retries,
                        rate_limit_retry_wait=rate_limit_retry_wait,
                    )
                else:
                    summaries = summarize_batch(
                        this_batch,
                        summary_store,
                        _budgeted_summarize_func,
                        update_mode,
                        LANGUAGE,
                        max_workers,
                        api_budget_tracker=_budget_tracker,
                    )
            except RateLimitAbort as exc:
                # Persist the partial results attached to the exception so
                # already-completed repos in this batch are not lost, then
                # stop the main loop so we don't keep hammering the
                # rate-limited API.
                print(f"[RATE_LIMIT] 主循环终止: {exc}")
                partial = getattr(exc, "results", None) or []
                for repo, summary in zip(this_batch, partial):
                    key = _repo_key(repo)
                    entry = build_repo_entry(repo, summary)
                    if key:
                        repo_summary_map[key] = entry
                if partial:
                    summary_store = merge_summary_store(summary_store, repo_summary_map)
                    save_json_atomic(summary_store, json_path)
                    print(f"[SAVE_PARTIAL] Saved {len(partial)} repos from interrupted batch")
                else:
                    # Even if this batch has no partial results, previous
                    # batches have been saved. Log the state for clarity.
                    print(f"[SAVE_PARTIAL] No partial results in this batch; "
                          f"{processed_repos}/{total_repos} repos already saved from prior batches")
                break

            for repo, summary in zip(this_batch, summaries):
                key = _repo_key(repo)
                entry = build_repo_entry(repo, summary)
                if key:
                    repo_summary_map[key] = entry

            summary_store = merge_summary_store(summary_store, repo_summary_map)
            save_json_atomic(summary_store, json_path)
            processed_repos += len(this_batch)
            print(
                f"[SAVE] Batch {batch_num} saved: {len(this_batch)} repos "
                f"({processed_repos}/{total_repos} total processed)"
            )

            # Waste ledger (P1-5): a batch where every entry carries an error
            # counts as a failed call. Too many in a row -> stop early and
            # keep the remaining RPD for tomorrow instead of burning it all.
            # (Budget-deferred batches are skipped here: they made no API
            # call, so they are neither progress nor waste.)
            batch_all_failed = bool(summaries) and all(
                isinstance(s, dict) and s.get("__last_error__") for s in summaries
            )
            batch_all_deferred = bool(summaries) and all(
                isinstance(s, dict)
                and str(s.get("__last_error__") or "").startswith("deferred")
                for s in summaries
            )
            if batch_all_deferred:
                print(f"[BUDGET] Batch {batch_num} fully deferred by budget; skipping remaining batches")
                break
            if batch_all_failed:
                consecutive_failed_batches += 1
                print(
                    f"[FAIL_STREAK] {consecutive_failed_batches}/{max_consecutive_failures} "
                    f"consecutive batches produced no content"
                )
                if 0 < max_consecutive_failures <= consecutive_failed_batches:
                    print(
                        f"[FAIL_STREAK] Aborting remaining batches after {consecutive_failed_batches} "
                        f"consecutive failures; progress is saved and failed repos will back off."
                    )
                    break
            else:
                consecutive_failed_batches = 0

        # Advance the round-robin cursor so the next run rotates cohorts.
        try:
            save_queue_state({"cursor": queue_cursor + 1})
        except Exception as exc:
            print(f"[QUEUE] failed to save queue state: {exc}")

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

        # Convergence report (P2-2) + run report consumed by the workflow
        # to decide whether the classify stage should run (P1-6).
        _invalid_after = sum(
            1 for e in (summary_store or {}).values()
            if isinstance(e, dict) and count_invalid_fields(e) > 0
        )
        _empty_summary_after = sum(
            1 for e in (summary_store or {}).values()
            if not (isinstance(e, dict) and (e.get("Summary") or "").strip())
        )
        new_valid_entries = max(0, _empty_summary_before - _empty_summary_after)
        print(
            f"[CONVERGENCE] after run: {_invalid_after} repos with >=1 invalid field "
            f"(was {_invalid_before}, fixed {_invalid_before - _invalid_after}), "
            f"{_empty_summary_after} with empty Summary (was {_empty_summary_before})"
        )
        try:
            repo_root_for_report = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(repo_root_for_report, "summarize_run_report.json"), "w", encoding="utf-8") as _rf:
                json.dump(
                    {
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "new_valid_entries": new_valid_entries,
                        "invalid_field_repos_before": _invalid_before,
                        "invalid_field_repos_after": _invalid_after,
                        "empty_summary_before": _empty_summary_before,
                        "empty_summary_after": _empty_summary_after,
                    },
                    _rf,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            print(f"[CONVERGENCE] failed to write run report: {exc}")

        # === Generate READMEs ===
        # 1. Content-classified README (README.md) - default
        # 2. Language-classified README (README_lang.md)

        repo_root = os.path.dirname(os.path.dirname(__file__))

        # Load content classification data
        categories_data = None
        categories_path = os.path.join(repo_root, "repo_categories.json")
        if os.path.exists(categories_path):
            try:
                import json as _json
                with open(categories_path, "r", encoding="utf-8") as _f:
                    categories_data = _json.load(_f)
            except Exception:
                categories_data = None

        # --- Content-classified README (README.md) ---
        if categories_data and categories_data.get("assignments"):
            print("\n[README] 生成内容分类 README (README.md)...")
            content_classified = classify_by_content(
                starred,
                categories_data.get("taxonomy", {}).get("categories", []),
                categories_data.get("assignments", []),
            )
            sort_by_count = config.get("content_sort_categories_by_count", True)

            content_lines: List[str] = []
            content_lines.extend(build_readme_header(LANGUAGE, GITHUB_USERNAME, api_name, len(starred), current_time))
            content_lines.extend(build_content_table_of_contents(content_classified, LANGUAGE, sort_by_count=sort_by_count))

            content_printed_repos: set = set()
            content_printed_cats: set = set()
            content_processed = 0

            # Sort categories: by count desc, Other last
            cat_items = list(content_classified.items())
            other_cats = [(k, v) for k, v in cat_items if k == "Other"]
            non_other_cats = [(k, v) for k, v in cat_items if k != "Other"]
            if sort_by_count:
                non_other_cats.sort(key=lambda x: -len(x[1]))
            sorted_cats = non_other_cats + other_cats

            for cat_name, repos in sorted_cats:
                section_lines, content_printed_repos, content_printed_cats, content_processed = build_content_repo_section(
                    cat_name,
                    repos,
                    LANGUAGE,
                    summary_store,
                    old_summaries,
                    rate_limit_delay,
                    content_printed_repos,
                    content_printed_cats,
                    content_processed,
                )
                content_lines.extend(section_lines)

            content_lines.extend(
                build_content_readme_footer(
                    content_processed,
                    len(content_classified),
                    current_time,
                    api_name,
                    (copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count),
                    LANGUAGE,
                )
            )

            readme_md_path = os.path.join(repo_root, "README.md")
            with open(readme_md_path, "w", encoding="utf-8") as f:
                f.write("".join(content_lines))
            print(f"✅ {readme_md_path} 已生成（内容分类），共 {content_processed} 个仓库。")
        else:
            print("\n[README] repo_categories.json 不存在或无分类数据，跳过内容分类 README 生成。")

        # --- Language-classified README (README_lang.md) ---
        print("\n[README] 生成语言分类 README (README_lang.md)...")
        lang_lines: List[str] = []
        lang_lines.extend(build_readme_header(LANGUAGE, GITHUB_USERNAME, api_name, len(starred), current_time))
        lang_lines.extend(build_table_of_contents(classified_to_process, LANGUAGE))

        lang_printed_repos: set = set()
        lang_printed_langs: set = set()
        lang_processed = 0

        for lang in sorted(classified_to_process.keys(), key=lambda x: -len(classified_to_process[x])):
            if lang in lang_printed_langs:
                continue
            repos = classified_to_process[lang]

            section_lines, lang_printed_repos, lang_printed_langs, lang_processed = build_repo_section(
                lang,
                repos,
                LANGUAGE,
                summary_store,
                old_summaries,
                rate_limit_delay,
                lang_printed_repos,
                lang_printed_langs,
                lang_processed,
            )
            lang_lines.extend(section_lines)

        lang_lines.extend(
            build_readme_footer(
                lang_processed,
                len(classified_to_process),
                current_time,
                api_name,
                (copilot_api_call_count, openrouter_api_call_count, gemini_api_call_count),
                LANGUAGE,
            )
        )

        readme_lang_path = os.path.join(repo_root, "README_lang.md")
        with open(readme_lang_path, "w", encoding="utf-8") as f:
            f.write("".join(lang_lines))
        print(f"✅ {readme_lang_path} 已生成（语言分类），共 {lang_processed} 个仓库。")

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
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the priority-queue preview (no API calls, no writes) and exit.",
    )
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

    if args.plan_only or (len(sys.argv) > 1 and sys.argv[1] == "--plan-only"):
        raise SystemExit(_run_plan_only())

    main()