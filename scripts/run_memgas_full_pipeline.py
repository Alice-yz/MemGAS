import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("longmemeval_s", "locomo10")
RETRIEVER = "contriever"
METHOD = "memgas"
TOPK = 3
NUM_SEEDNODES = 15
MEM_THRESHOLD = 30
N_COMPONENTS = 2
DAMPING = 0.1
TEMP = 0.2
LLM_MODEL = "gpt-4o-mini"
LLM_CONCURRENCY = 4
EMB_BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full MemGAS reproduce pipeline with all LLM API calls fixed to gpt-4o-mini."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to run. Defaults to longmemeval_s and locomo10.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Profiling run id. Defaults to memgas_full_<timestamp>.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove this run's dataset-specific caches/outputs before running.",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip data preprocessing and use files already under data/process_data/.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and cleanup targets without running them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subprocesses. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_step(name, cmd, cwd, dry_run=False):
    print(f"\n=== {name} ===", flush=True)
    print(f"cwd: {cwd}", flush=True)
    print(quote_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def dataset_prefix_for_multigran(dataset):
    if "longmemeval" in dataset:
        return "longmemeval"
    return dataset


def generation_file(dataset):
    return f"{dataset}-{RETRIEVER}-{METHOD}_filter-{LLM_MODEL}-topk_{TOPK}.jsonl"


def judge_file(dataset):
    return REPO_ROOT / "generation_logs" / "llm_judge_single" / f"{generation_file(dataset)}-{LLM_MODEL}_judge.jsonl"


def summary_file(run_id, dataset):
    return REPO_ROOT / "profiling_logs" / f"{run_id}-{dataset}-summary.json"


def profile_file(run_id):
    return REPO_ROOT / "profiling_logs" / f"{run_id}.jsonl"


def maybe_unlink(path, dry_run=False):
    if dry_run:
        print(f"[dry-run] remove if exists: {path}", flush=True)
        return
    if path.exists():
        path.unlink()
        print(f"removed: {path}", flush=True)


def cleanup_for_run(run_id, datasets, dry_run=False):
    maybe_unlink(profile_file(run_id), dry_run=dry_run)
    for dataset in datasets:
        multigran_prefix = dataset_prefix_for_multigran(dataset)
        for level in ("summary_level", "keyword_level"):
            maybe_unlink(
                REPO_ROOT / "multi_granularity_logs" / f"{multigran_prefix}-{level}.jsonl",
                dry_run=dry_run,
            )
        maybe_unlink(
            REPO_ROOT / "data" / "process_embs" / f"{dataset}-{RETRIEVER}-emb.pt",
            dry_run=dry_run,
        )
        maybe_unlink(
            REPO_ROOT
            / "graph_cache"
            / f"graph-{dataset}-{RETRIEVER}-{MEM_THRESHOLD}-{N_COMPONENTS}.pt",
            dry_run=dry_run,
        )
        maybe_unlink(
            REPO_ROOT / "retrieval_logs" / f"{dataset}-{RETRIEVER}-{METHOD}.jsonl",
            dry_run=dry_run,
        )
        maybe_unlink(REPO_ROOT / "generation_logs" / generation_file(dataset), dry_run=dry_run)
        maybe_unlink(REPO_ROOT / "generation_logs" / "metrics" / generation_file(dataset), dry_run=dry_run)
        maybe_unlink(judge_file(dataset), dry_run=dry_run)
        maybe_unlink(summary_file(run_id, dataset), dry_run=dry_run)


def ensure_inputs(datasets, skip_preprocess):
    if skip_preprocess:
        required = {
            "longmemeval_s": REPO_ROOT / "data" / "process_data" / "longmemeval_s.json",
            "locomo10": REPO_ROOT / "data" / "process_data" / "locomo10.json",
        }
    else:
        required = {
            "longmemeval_s": REPO_ROOT / "data" / "origin_data" / "longmemeval_s",
            "locomo10": REPO_ROOT / "data" / "origin_data" / "locomo10.json",
        }
    missing = [str(required[dataset]) for dataset in datasets if not required[dataset].exists()]
    if missing:
        kind = "processed" if skip_preprocess else "raw"
        raise FileNotFoundError(f"Missing {kind} data files: {missing}")


def preprocess_dataset(dataset, python, dry_run=False):
    if not dry_run:
        (REPO_ROOT / "data" / "process_data").mkdir(exist_ok=True)
    function_name = {
        "longmemeval_s": "process_longmemeval",
        "locomo10": "process_locomo10",
    }[dataset]
    run_step(
        f"Preprocess {dataset}",
        [
            python,
            "-c",
            f"from dataprocess import {function_name}; {function_name}()",
        ],
        cwd=REPO_ROOT / "data",
        dry_run=dry_run,
    )


def build_memory(dataset, run_id, python, dry_run=False):
    run_step(
        f"Build multi-granularity memory for {dataset}",
        [
            python,
            "multigran_generation.py",
            "--dataset",
            dataset,
            "--level",
            "summary_level",
            "--level",
            "keyword_level",
            "--run_id",
            run_id,
        ],
        cwd=REPO_ROOT / "src" / "construct",
        dry_run=dry_run,
    )


def run_retrieval(dataset, run_id, python, dry_run=False):
    run_step(
        f"Run MemGAS retrieval for {dataset}",
        [
            python,
            "run_retrieval.py",
            "--dataset",
            dataset,
            "--retriever",
            RETRIEVER,
            "--method",
            METHOD,
            "--num_seednodes",
            str(NUM_SEEDNODES),
            "--mem_threshold",
            str(MEM_THRESHOLD),
            "--n_components",
            str(N_COMPONENTS),
            "--damping",
            str(DAMPING),
            "--temp",
            str(TEMP),
            "--run_id",
            run_id,
            "--emb_batch_size",
            str(EMB_BATCH_SIZE),
        ],
        cwd=REPO_ROOT / "src" / "retrieval",
        dry_run=dry_run,
    )


def generate_answers(dataset, run_id, python, dry_run=False):
    run_step(
        f"Generate answers for {dataset}",
        [
            python,
            "generation_multigran.py",
            "--dataset",
            dataset,
            "--retriever",
            RETRIEVER,
            "--model_name_or_path",
            LLM_MODEL,
            "--topk",
            str(TOPK),
            "--method",
            METHOD,
            "--run_id",
            run_id,
            "--llm_concurrency",
            str(LLM_CONCURRENCY),
        ],
        cwd=REPO_ROOT / "src" / "generation",
        dry_run=dry_run,
    )


def judge_answers(dataset, run_id, python, dry_run=False):
    run_step(
        f"Judge answers for {dataset}",
        [
            python,
            "llm_judge_single.py",
            "--dataset",
            dataset,
            "--model_name_or_path",
            LLM_MODEL,
            "--eval_file",
            generation_file(dataset),
            "--run_id",
            run_id,
            "--overwrite",
            "--llm_concurrency",
            str(LLM_CONCURRENCY),
        ],
        cwd=REPO_ROOT / "src" / "evaluation",
        dry_run=dry_run,
    )


def aggregate_results(dataset, run_id, python, dry_run=False):
    out_path = summary_file(run_id, dataset)
    run_step(
        f"Aggregate profiling for {dataset}",
        [
            python,
            "src/evaluation/aggregate_profiling.py",
            "--run_id",
            run_id,
            "--dataset",
            dataset,
            "--judge_file",
            str(judge_file(dataset).relative_to(REPO_ROOT)),
            "--output",
            str(out_path.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        dry_run=dry_run,
    )
    return out_path


def print_summary(path, dataset):
    if not path.exists():
        print(f"{dataset}: summary not found at {path}", flush=True)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    memgas = data.get("memgas_included_cost", {})
    accuracy = (data.get("judge_accuracy") or {}).get("accuracy")
    print(f"\n--- {dataset} final summary ---", flush=True)
    print(f"summary_file: {path}", flush=True)
    print(f"accuracy: {accuracy}", flush=True)
    print(f"memgas_wall_time_s: {memgas.get('wall_time_s')}", flush=True)
    print(f"memgas_prompt_tokens: {memgas.get('prompt_tokens')}", flush=True)
    print(f"memgas_completion_tokens: {memgas.get('completion_tokens')}", flush=True)
    print(f"memgas_total_tokens: {memgas.get('total_tokens')}", flush=True)
    missing = data.get("usage_missing_stages") or []
    if missing:
        print(f"WARNING: usage_missing_stages={missing}", flush=True)


def main():
    args = parse_args()
    run_id = args.run_id or f"memgas_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("MemGAS full pipeline configuration", flush=True)
    print(f"run_id: {run_id}", flush=True)
    print(f"datasets: {', '.join(args.datasets)}", flush=True)
    print(f"retriever: {RETRIEVER}", flush=True)
    print(f"method: {METHOD}", flush=True)
    print(f"topk: {TOPK}", flush=True)
    print(f"num_seednodes: {NUM_SEEDNODES}", flush=True)
    print(f"mem_threshold: {MEM_THRESHOLD}", flush=True)
    print(f"n_components: {N_COMPONENTS}", flush=True)
    print(f"damping: {DAMPING}", flush=True)
    print(f"temp: {TEMP}", flush=True)
    print(f"all_llm_api_model: {LLM_MODEL}", flush=True)
    print("answer_api_temperature: 0 (set in src/generation/async_llm.py)", flush=True)
    print(f"llm_concurrency: {LLM_CONCURRENCY}", flush=True)
    print(f"emb_batch_size: {EMB_BATCH_SIZE}", flush=True)

    if args.dry_run:
        print("dry_run: skip input existence checks", flush=True)
    else:
        ensure_inputs(args.datasets, args.skip_preprocess)
    if args.fresh:
        cleanup_for_run(run_id, args.datasets, dry_run=args.dry_run)

    summary_paths = []
    try:
        for dataset in args.datasets:
            if not args.skip_preprocess:
                preprocess_dataset(dataset, args.python, dry_run=args.dry_run)
            build_memory(dataset, run_id, args.python, dry_run=args.dry_run)
            run_retrieval(dataset, run_id, args.python, dry_run=args.dry_run)
            generate_answers(dataset, run_id, args.python, dry_run=args.dry_run)
            judge_answers(dataset, run_id, args.python, dry_run=args.dry_run)
            summary_paths.append((dataset, aggregate_results(dataset, run_id, args.python, dry_run=args.dry_run)))
    except subprocess.CalledProcessError as err:
        print(f"\nPipeline failed with exit code {err.returncode}: {quote_cmd(err.cmd)}", file=sys.stderr)
        raise

    print("\nPipeline completed.", flush=True)
    print(f"profile_file: {profile_file(run_id)}", flush=True)
    for dataset, path in summary_paths:
        if args.dry_run:
            print(f"{dataset} summary_file: {path}", flush=True)
        else:
            print_summary(path, dataset)


if __name__ == "__main__":
    main()
