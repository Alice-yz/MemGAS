import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../")))

from generation.prompt_templates import infer_dataset_from_filename, parse_judge_score
from src.profiling_utils import get_profile_path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_records(records):
    return {
        "wall_time_s": sum(float(row.get("wall_time_s") or 0) for row in records),
        "api_latency_s": sum(float(row.get("api_latency_s") or 0) for row in records),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in records),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in records),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in records),
        "call_count": sum(int(row.get("call_count") or 0) for row in records),
    }


def calculate_accuracy(judge_file, dataset):
    rows = read_jsonl(judge_file)
    scores = []
    for row in rows:
        score = row.get("llm_judge_score")
        if score is None:
            score = parse_judge_score(dataset, row.get("llm_judge_single", ""))
        scores.append(int(score))
    if not scores:
        return None
    return {
        "correct": sum(scores),
        "total": len(scores),
        "accuracy": round(sum(scores) / len(scores) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate MemGAS profiling logs.")
    parser.add_argument("--profile_file", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--judge_file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    profile_file = args.profile_file or get_profile_path(run_id=args.run_id)
    all_records = read_jsonl(profile_file)
    dataset = args.dataset
    if dataset is None and args.judge_file:
        dataset = infer_dataset_from_filename(args.judge_file)
    records = [row for row in all_records if row.get("dataset") == dataset] if dataset else all_records
    included = [row for row in records if row.get("included_in_memgas_cost")]
    excluded = [row for row in records if not row.get("included_in_memgas_cost")]
    output = {
        "profile_file": profile_file,
        "dataset": dataset,
        "memgas_included_cost": summarize_records(included),
        "excluded_llm_cost": summarize_records(excluded),
        "usage_missing_stages": [
            row.get("stage")
            for row in records
            if int(row.get("call_count") or 0) > 0 and row.get("usage_available") is False
        ],
        "stage_breakdown": {
            stage: summarize_records([row for row in records if row.get("stage") == stage])
            for stage in sorted({row.get("stage") for row in records if row.get("stage")})
        },
    }
    if args.judge_file:
        if dataset is None:
            raise ValueError("Unable to infer dataset from --judge_file. Please pass --dataset.")
        output["judge_accuracy"] = calculate_accuracy(args.judge_file, dataset)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
