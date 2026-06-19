import json
import os
from typing import Any, Dict

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH_JSON = os.path.join(_REPO_ROOT, "config.json")
CONFIG_PATH_YAML = os.path.join(_REPO_ROOT, "config.yaml")


def load_config() -> Dict[str, Any]:
    print(f"DEBUG load_config: CONFIG_PATH_YAML={CONFIG_PATH_YAML}")
    try:
        with open(CONFIG_PATH_YAML, "r", encoding="utf-8") as f:
            raw = f.read()
            print(f"DEBUG load_config: raw yaml length={len(raw)}, first 200 chars={repr(raw[:200])}")
            data = yaml.safe_load(raw)
            print(f"DEBUG load_config: yaml.safe_load result keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
            if isinstance(data, dict):
                return data
    except FileNotFoundError as e:
        print(f"DEBUG load_config: FileNotFoundError: {e}")
        pass
    except Exception as e:
        print(f"DEBUG load_config: Exception: {e}")
        pass

    print(f"DEBUG load_config: CONFIG_PATH_JSON={CONFIG_PATH_JSON}")
    try:
        with open(CONFIG_PATH_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"DEBUG load_config: json.load result keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
            if isinstance(data, dict):
                return data
    except FileNotFoundError as e:
        print(f"DEBUG load_config: FileNotFoundError: {e}")
        pass
    except Exception as e:
        print(f"DEBUG load_config: Exception: {e}")
        pass

    return {"language": "zh"}


def get_int_config(config: Dict[str, Any], key: str, default: int) -> int:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)


def get_float_config(config: Dict[str, Any], key: str, default: float) -> float:
    try:
        val = config.get(key, default) if isinstance(config, dict) else default
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def env_truthy(name: str) -> bool:
    try:
        return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return False


def normalize_update_mode(mode: Any) -> str:
    try:
        s = str(mode or "").strip().lower()
    except Exception:
        s = ""
    s = s.replace("-", "_")
    if s in {"missing", "missingonly", "missing_only", "incremental"}:
        return "missing_only"
    if s in {"all", "full"}:
        return "all"
    if s in {"force_all", "forceall", "force"}:
        return "force_all"
    return "missing_only"


def resolve_update_mode(config: Dict[str, Any]) -> str:
    return normalize_update_mode(
        os.environ.get("MYGITSTAR_UPDATE_MODE")
        or (config.get("update_mode") if isinstance(config, dict) else None)
        or "missing_only"
    )


def get_model_choice(config: Dict[str, Any]) -> str:
    try:
        val = os.environ.get("MYGITSTAR_MODEL_CHOICE")
        if val is not None and str(val).strip():
            return str(val).strip().lower()
        return (config.get("model_choice", "copilot") if isinstance(config, dict) else "copilot").strip().lower()
    except Exception:
        return "copilot"
