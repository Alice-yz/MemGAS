
import argparse
import json
import os
import sys
from time import perf_counter

from metrics import evaluate_match, evaluate_sim
from tqdm import tqdm
from async_llm import run_async
from prompt_templates import build_answer_messages
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, repo_root)
from src.profiling_utils import append_profile_record, sum_usage

# locomo10 longmemeval_s longmemeval_m LongMTBench+
# python generation_multigran.py --dataset longmemeval_m --retriever contriever --model_name_or_path gpt-4o-mini --topk 3 --method memgas

parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--retriever', type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini")
parser.add_argument('--topk', type=int, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--run_id', type=str, default=None)
parser.add_argument('--profile_path', type=str, default=None)
parser.add_argument('--llm_concurrency', type=int, default=4)

args = parser.parse_args()
os.makedirs("../../generation_logs/", exist_ok=True)
save_path = f'../../generation_logs/{args.dataset}-{args.retriever}-{args.method}_filter-{args.model_name_or_path}-topk_{args.topk}.jsonl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)


retrieved_data = []
with open(f'../../retrieval_logs/{args.dataset}-{args.retriever}-{args.method}.jsonl', "r") as f:
    for line in f.readlines():
        retrieved_data.append(json.loads(line.strip()))
in_data = json.load(open(f'../../data/process_data/{args.dataset}.json'))

################ session based
conv2sessions = {}
for entry in in_data:
    ids2session = {k:v for k,v in zip(entry['sessions_ids'],entry['sessions'])}
    conv2sessions[entry['conversation_id']] = ids2session
################

################ other granularity
def get_multigran(level):
    conv2multigran = {}
    summ_dict = {}
    if 'longmemeval' in args.dataset:
        multi_gran_path = f'../../multi_granularity_logs/longmemeval-{level}.jsonl'
    else:
        multi_gran_path = f'../../multi_granularity_logs/{args.dataset}-{level}.jsonl'
    with open(multi_gran_path, "r") as f:
        for line in f.readlines():
            summ_dict.update(json.loads(line.strip()))
    for entry in in_data:
        ids2summary = {}
        conv_id = entry['conversation_id']
        for sessid in entry['sessions_ids']:
            if 'longmemeval' in args.dataset:
                summ_text = summ_dict[f'{sessid}']
            else:
                summ_text = summ_dict[f'convid-{str(conv_id)}-sessid-{sessid}']
            ids2summary[sessid] = summ_text
        conv2multigran[conv_id] = ids2summary
    return conv2multigran
conv2summary = get_multigran('summary_level')
conv2keyword = get_multigran('keyword_level')


########################
# results = []
# if os.path.exists(save_path):
#     with open(save_path, "r") as f:
#         for line in f.readlines():
#             sample = json.loads(line.strip())
#             results.append(sample)
######## 并行 



PROMPT_Multigran = """
You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular information.
Filter the History Dialogs, summaries, and keywords to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
Answer:
"""

async_prompts = []
for idx, sample in enumerate(tqdm(retrieved_data)):
    ids2session = conv2sessions[sample["conversation_id"]]
    ids2summary = conv2summary[sample["conversation_id"]]
    ids2keyword = conv2keyword[sample["conversation_id"]]
    retrieved_texts = ""
    for retrieved_sess in sample['retrieval_results']['ranked_items'][:args.topk]:
        session = ids2session[retrieved_sess['corpus_id']]
        summary = ids2summary[retrieved_sess['corpus_id']]
        keyword = ids2keyword[retrieved_sess['corpus_id']]
        retrieved_texts += f"\n### Session Date: {retrieved_sess['timestamp']}\nSession Content:\n{session}\n\nSession Summary:\n{summary}\nSession Keyword:\n{keyword}\n"
    prompt = PROMPT_Multigran.format(retrieved_texts=retrieved_texts, question=sample["question"], question_date=sample["question_date"])
    async_prompts.append(prompt)

stage_started_at = perf_counter()
filter_results = asyncio.run(
    run_async(
        async_prompts,
        args.model_name_or_path,
        return_usage=True,
        max_concurrency=args.llm_concurrency,
    )
)
filter_wall_time_s = perf_counter() - stage_started_at
filter_usage = sum_usage([result.get("usage") for result in filter_results if isinstance(result, dict)])
append_profile_record(
    dataset=args.dataset,
    stage="memgas_filter_llm",
    included_in_memgas_cost=True,
    wall_time_s=filter_wall_time_s,
    api_latency_s=sum(result.get("api_latency_s", 0) for result in filter_results if isinstance(result, dict)),
    call_count=len(filter_results),
    run_id=args.run_id,
    profile_path=args.profile_path,
    usage_available=all(isinstance(result, dict) and bool(result.get("usage")) for result in filter_results),
    **filter_usage,
)

async_prompts = []
for idx, sample in enumerate(tqdm(retrieved_data)):
    filtered_memory = filter_results[idx].get("content", "") if isinstance(filter_results[idx], dict) else filter_results[idx]
    async_prompts.append(build_answer_messages(args.dataset, sample, filtered_memory))

stage_started_at = perf_counter()
answer_results = asyncio.run(
    run_async(
        async_prompts,
        args.model_name_or_path,
        return_usage=True,
        max_concurrency=args.llm_concurrency,
    )
)
answer_wall_time_s = perf_counter() - stage_started_at
answer_usage = sum_usage([result.get("usage") for result in answer_results if isinstance(result, dict)])
append_profile_record(
    dataset=args.dataset,
    stage="final_answer_llm",
    included_in_memgas_cost=False,
    wall_time_s=answer_wall_time_s,
    api_latency_s=sum(result.get("api_latency_s", 0) for result in answer_results if isinstance(result, dict)),
    call_count=len(answer_results),
    run_id=args.run_id,
    profile_path=args.profile_path,
    usage_available=all(isinstance(result, dict) and bool(result.get("usage")) for result in answer_results),
    **answer_usage,
)

results = []
for sample, response in zip(retrieved_data, answer_results):
    sample["response"] = response.get("content", "") if isinstance(response, dict) else response
    results.append(sample)
with open(save_path, "w", encoding="utf-8") as f:
    f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])


def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        if ans is not None:
            ans = ans.strip(strip_word)
        else:
            ans = ""
        
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans

print('Calculating metrics')
pred_all = []
for res in results:
    ans = get_answer(res["response"])
    pred_all.append(ans)

answer_all = []
for res in results:
    answer_all.append(str(res["answer"]))

metrics = evaluate_sim(pred_all, answer_all, truncate_pred=False)
metrics.update(evaluate_match(pred_all, answer_all, truncate_pred=False))
print(metrics)
metrics_dir = os.path.join(os.path.dirname(save_path), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
with open(
    os.path.join(
        metrics_dir, os.path.basename(save_path).replace("answer", "metrics")
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f)
