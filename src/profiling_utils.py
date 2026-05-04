import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PROFILE_DIR = os.path.join(REPO_ROOT, "profiling_logs")


def get_profile_path(run_id=None, profile_path=None):
    if profile_path:
        return profile_path
    resolved_run_id = run_id or os.environ.get("MEMGAS_RUN_ID") or "default"
    return os.path.join(DEFAULT_PROFILE_DIR, f"{resolved_run_id}.jsonl")


def usage_to_dict(usage):
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def sum_usage(usages):
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        cur = usage_to_dict(usage)
        for key in total:
            total[key] += cur[key]
    return total


def append_profile_record(
    dataset,
    stage,
    included_in_memgas_cost,
    wall_time_s,
    run_id=None,
    profile_path=None,
    api_latency_s=0,
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    call_count=0,
    cache_hit=False,
    **extra,
):
    path = get_profile_path(run_id=run_id, profile_path=profile_path)
    profile_dir = os.path.dirname(path)
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "stage": stage,
        "included_in_memgas_cost": bool(included_in_memgas_cost),
        "wall_time_s": float(wall_time_s or 0),
        "api_latency_s": float(api_latency_s or 0),
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "call_count": int(call_count or 0),
        "cache_hit": bool(cache_hit),
    }
    record.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@contextmanager
def elapsed_timer():
    start = time.perf_counter()

    def elapsed():
        return time.perf_counter() - start

    yield elapsed
