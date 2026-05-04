import json
import re
from string import Template


LOCOMO_JUDGE_PROMPT_TEMPLATE = Template(
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: "
    "(1) a question (posed by one user to another user), "
    "(2) a 'gold' (ground truth) answer, "
    "(3) a generated answer "
    "which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
    "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
    "Special rule for EMPTY gold answer: If Gold answer is an empty string, then the answer is CORRECT ONLY if the generated answer clearly indicates the question cannot be answered based on the provided information (e.g., insufficient info / cannot determine / some other information is given but the asked information is not). Otherwise WRONG.\n\n"
    "Now it's time for the real question:\n"
    "Question: $question\n"
    "Gold answer: $golden_answers\n"
    "Generated answer: $prediction\n\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
    "Just return the label CORRECT or WRONG in a json format with the key as 'label'."
)


DEFAULT_JUDGE_PROMPT_TEMPLATE = (
    "I will give you a question, a reference answer, and a response from a model. Please answer [[yes]] if the response contains the reference answer. Otherwise, answer [[no]]. \n"
    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the reference answer, you should also answer [[yes]]. If the response only contains a subset of the information required by the answer, answer [[no]]. \n\n"
    "[User Question]\n{question}\n\n"
    "[The Start of Reference Answer]\n{answer}\n[The End of Reference Answer]\n\n"
    "[The Start of Model's Response]\n{response}\n[The End of Model's Response]\n\n"
    "Is the model response correct? Answer [[yes]] or [[no]] only."
)


def is_locomo(dataset):
    return dataset == "locomo10"


def is_longmemeval(dataset):
    return "longmemeval" in dataset.lower()


def infer_dataset_from_filename(path):
    name = path.split("/")[-1]
    for dataset in ("longmemeval_s", "longmemeval_m", "locomo10", "LongMTBench+"):
        if name.startswith(dataset):
            return dataset
    return None


def _category(sample):
    try:
        return int(sample.get("question_type"))
    except (TypeError, ValueError):
        return sample.get("question_type")


def build_answer_messages(dataset, sample, retrieved_texts):
    question = sample["question"]
    if is_locomo(dataset):
        if _category(sample) == 3:
            content = (
                f"Memories:\n{str(retrieved_texts)}\n\n"
                "Based on the above memories, write an answer in the form of a short phrase for the following question, not a sentence. "
                "The question may need you to analyze and infer the answer from the summary.\n"
                f"Question: {question}\n"
                "Short answer:"
            )
        else:
            content = (
                f"Memories:\n{str(retrieved_texts)}\n\n"
                "Based on the above memories, write an answer in the form of a short phrase for the following question, not a sentence. "
                "Answer with exact words from the memories whenever possible. For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. "
                "For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" "
                "Only provide one year, date, or time, without any extra responses. If the question is about the duration, answer in the form of several years, months, or days.\n"
                f"Question: {question}\n"
                "Short answer:"
            )
        return [{"role": "user", "content": content}]

    if is_longmemeval(dataset):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"Question time:{sample.get('question_date')} and question:{question}\n"
                    f"Please answer the question based on the following memories: {retrieved_texts}"
                ),
            },
        ]

    content = (
        "You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, "
        "then generate one concise, coherent and helpful response for the Question.\n\n"
        f"History Dialogs: {retrieved_texts}\n\n"
        f"Question Date: {sample.get('question_date')}\n"
        f"Question: {question}\n"
    )
    return [{"role": "user", "content": content}]


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unsupported LongMemEval task: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def build_judge_prompt(dataset, sample):
    question = sample["question"]
    answer = sample.get("answer", "")
    response = sample.get("response", "")
    if is_locomo(dataset):
        return LOCOMO_JUDGE_PROMPT_TEMPLATE.safe_substitute(
            question=question,
            golden_answers=answer,
            prediction=response,
        )
    if is_longmemeval(dataset):
        return get_anscheck_prompt(
            sample.get("question_type"),
            question,
            answer,
            response,
            abstention="_abs" in str(sample.get("conversation_id", "")),
        )
    return DEFAULT_JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer, response=response)


def judge_prompt_version(dataset):
    if is_locomo(dataset):
        return "locomo_json_correct_wrong_v1"
    if is_longmemeval(dataset):
        return "longmemeval_anscheck_v1"
    return "default_bracket_yes_no_v1"


def _extract_json_label(response):
    text = (response or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    label = parsed.get("label")
    if label is None:
        return None
    return str(label).strip().upper()


def parse_judge_score(dataset, response):
    text = (response or "").strip()
    lowered = text.lower()
    if is_locomo(dataset):
        label = _extract_json_label(response)
        if label == "CORRECT":
            return 1
        if label == "WRONG":
            return 0
        has_correct = "correct" in lowered
        has_wrong = "wrong" in lowered
        return 1 if has_correct and not has_wrong else 0
    if is_longmemeval(dataset):
        has_yes = re.search(r"\byes\b", lowered) is not None
        has_no = re.search(r"\bno\b", lowered) is not None
        return 1 if has_yes and not has_no else 0
    if "[[yes]]" in lowered and "[[no]]" not in lowered:
        return 1
    return 0
