
import argparse
import json
import os
from tqdm import tqdm
import os
import sys
from time import perf_counter
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_path)
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, repo_root)
import asyncio
from generation.async_llm import run_async
from generation.prompt_templates import (
    build_judge_prompt,
    infer_dataset_from_filename,
    judge_prompt_version,
    parse_judge_score,
)
from src.profiling_utils import append_profile_record, sum_usage

# locomo10 longmemeval_s longmemeval_m LongMTBench+
# memgas_filter
# python llm_judge_single.py --model_name_or_path gpt-4o --eval_file  longmemeval_s-contriever-memgas_filter-gpt-4o-mini-topk_3.jsonl


parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument('--eval_file', type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="gpt-4o")
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--run_id', type=str, default=None)
parser.add_argument('--profile_path', type=str, default=None)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--llm_concurrency', type=int, default=4)
args = parser.parse_args()
dataset = args.dataset or infer_dataset_from_filename(args.eval_file)
if dataset is None:
    raise ValueError("Unable to infer dataset from --eval_file. Please pass --dataset.")

g_path = f'../../generation_logs/{args.eval_file}'

file_name = os.path.basename(g_path)
save_path = f'../../generation_logs/llm_judge_single/{file_name}-{args.model_name_or_path}_judge.jsonl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
prompt_version = judge_prompt_version(dataset)

def calculate_single(path, dataset):
    results = []
    with open(path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            score = sample.get("llm_judge_score")
            if score is None:
                score = parse_judge_score(dataset, sample.get('llm_judge_single', ''))
            results.append(int(score))
    print("llm_judge_single Acc: ",round(sum(results)/len(results)*100,2))

def is_current_judge_file(path, expected_version):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            return sample.get("judge_prompt_version") == expected_version
    return False

if os.path.exists(save_path) and not args.overwrite and is_current_judge_file(save_path, prompt_version):
    calculate_single(save_path, dataset)
else:
    g_results = []
    async_prompts = []
    with open(g_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            g_results.append(sample)
            prompt = build_judge_prompt(dataset, sample)
            async_prompts.append(prompt)

    eval_results = []
    stage_started_at = perf_counter()
    async_responses = asyncio.run(
        run_async(
            async_prompts,
            args.model_name_or_path,
            return_usage=True,
            max_concurrency=args.llm_concurrency,
        )
    )
    judge_wall_time_s = perf_counter() - stage_started_at
    judge_usage = sum_usage([result.get("usage") for result in async_responses if isinstance(result, dict)])
    append_profile_record(
        dataset=dataset,
        stage="judge_llm",
        included_in_memgas_cost=False,
        wall_time_s=judge_wall_time_s,
        api_latency_s=sum(result.get("api_latency_s", 0) for result in async_responses if isinstance(result, dict)),
        call_count=len(async_responses),
        run_id=args.run_id,
        profile_path=args.profile_path,
        usage_available=all(isinstance(result, dict) and bool(result.get("usage")) for result in async_responses),
        **judge_usage,
    )
    for sample, response in zip(g_results,async_responses):
        judge_response = response.get("content", "") if isinstance(response, dict) else response
        sample["llm_judge_single"] = judge_response
        sample["llm_judge_score"] = parse_judge_score(dataset, judge_response)
        sample["judge_prompt_version"] = prompt_version
        sample.pop("retrieval_results", None)
        eval_results.append(sample)

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in eval_results])
    calculate_single(save_path, dataset)
