import argparse
import csv
import os
import json
import random
import re
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NER_TASKS = {"bc4chem", "bc5chem", "bc5disease", "species800"}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="qwen-4B")
parser.add_argument("--api_base", default="http://localhost:8000/v1")
parser.add_argument("--benchmark_task", required=True)
parser.add_argument("--data_file", required=True)
parser.add_argument("--max_samples", type=int, default=500)
parser.add_argument("--max_tokens", type=int, default=16384)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--presence_penalty", type=float, default=1.5)
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--enable_thinking", action="store_true", default=True)
parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
parser.add_argument("--thinking_budget", type=int, default=8192)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--min_p", type=float, default=0.0)
parser.add_argument("--workers", type=int, default=32)
args = parser.parse_args()

is_ner = any(t in args.benchmark_task.lower() for t in NER_TASKS)

client = OpenAI(base_url=args.api_base, api_key="dummy")
os.makedirs("./prediction_results", exist_ok=True)
SAVE_PATH = f"./prediction_results/eval_{args.benchmark_task}_{args.model_name}.csv"

with open(args.data_file, "r") as f:
    data = json.load(f)

definition = data["Definition"][0]
instances  = data["Instances"]
pos_examples = data.get("Positive Examples", [])
few_shot = ""
for ex in pos_examples[:3]:
    few_shot += f"Input:\n{ex['input']}\nOutput:\nAnswer: {ex['output']}\n\n"

if is_ner:
    format_instruction = (
        "\n\nCritical rules you must follow:"
        "\n1. Extract entity names EXACTLY as they appear in the input text. Do not expand abbreviations, do not substitute full names. If the text says 'Mg', output 'Mg', not 'Magnesium'."
        "\n2. If the sentence contains no relevant entity, output exactly: There is no related entity."
        "\n3. Do not hallucinate entities. Only extract what is explicitly present in the input text."
        "\n4. Do not include proteins, genes, or non-chemical biological terms unless they are also chemical compounds."
        "\n5. Do not explain your reasoning. Do not write any sentences. Output only the final answer line."
        "\n6. Your entire response after thinking must be one single line starting with 'Answer:' and nothing else."
        "\nAnswer: <comma-separated entity names as they appear in the text, or 'There is no related entity.' if none>"
    )
else:
    format_instruction = "\nAfter your reasoning, you must end your response with exactly:\nAnswer: <answer>"

system_prompt = (few_shot + definition + format_instruction + "\nPlease learn from the few-shot cases above.") if few_shot else (definition + format_instruction)

random.seed(42)
if len(instances) > args.max_samples:
    instances = random.sample(instances, args.max_samples)

print(f"Task:             {args.benchmark_task}")
print(f"Mode:             {'NER' if is_ner else 'Classification'}")
print(f"Samples:          {len(instances)}")
print(f"Max tokens:       {args.max_tokens}")
print(f"Temperature:      {args.temperature}")
print(f"Top-p:            {args.top_p}")
print(f"Top-k:            {args.top_k}")
print(f"Presence penalty: {args.presence_penalty}")
print(f"Thinking:         {args.enable_thinking}")
print(f"Thinking budget:  {args.thinking_budget}")
print(f"Repetition pen.:  {args.repetition_penalty}")
print(f"Min-p:            {args.min_p}")
print(f"Workers:          {args.workers}")
print(f"Prompt preview:   {system_prompt[:300]}\n")
print("-" * 60)


def parse_answer(raw: str) -> str:
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.count("!") > 10 or raw.count("?") > 10:
        return "PARSE_ERROR: repetition loop detected"
    match = re.search(r"Answer:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().rstrip(".")
    return raw.strip()


def normalize_gt(gt: str) -> str:
    match = re.search(r"(?:diagnosis result is|diagnosis is|treatment planning is:?)\s*(.+)", gt, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    return gt.strip()


def parse_entities(text: str) -> set:
    text = text.strip().lower()
    if "no related" in text or text == "none":
        return set()
    return {e.strip().rstrip(".") for e in re.split(r",|;", text) if e.strip()}


def token_f1(pred_set: set, gt_set: set) -> tuple:
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gt_set or not pred_set:
        return 0.0, 0.0, 0.0
    pred_tokens = set(" ".join(pred_set).split())
    gt_tokens   = set(" ".join(gt_set).split())
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall    = tp / len(gt_tokens)   if gt_tokens   else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run_instance(inst, idx):
    user_msg = f"Input:\n{inst['input']}\nOutput:"
    gt_raw = inst["output"] if isinstance(inst["output"], str) else inst["output"][0]

    effective_thinking = args.enable_thinking and not is_ner
    extra = {"top_k": args.top_k, "chat_template_kwargs": {"enable_thinking": effective_thinking}}
    if effective_thinking:
        extra["thinking"] = {"type": "enabled", "budget_tokens": args.thinking_budget}

    try:
        resp = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg}
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            extra_body={**extra, "repetition_penalty": args.repetition_penalty, "min_p": args.min_p},
        )
        content = resp.choices[0].message.content
        if content is None:
            raw = pred_text = "ERROR: model returned null content (thinking consumed all tokens)"
        else:
            raw = content.strip()
            pred_text = parse_answer(raw)
    except Exception as e:
        raw = pred_text = f"ERROR: {e}"

    result = {
        "idx": idx,
        "row": idx + 1,
        "task_id": inst.get("id", idx),
        "input": inst["input"][:200],
        "gt": gt_raw,
        "raw": raw[:300],
        "pred": pred_text[:200],
    }

    if is_ner:
        pred_entities = parse_entities(pred_text)
        gt_entities   = parse_entities(gt_raw)
        p, r, f1 = token_f1(pred_entities, gt_entities)
        result.update({"precision": p, "recall": r, "f1": f1})
    else:
        gt_norm = normalize_gt(gt_raw)
        is_correct = int(pred_text.strip().lower() in gt_norm.strip().lower() or gt_norm.strip().lower() in pred_text.strip().lower())
        result.update({"gt_norm": gt_norm, "correct": is_correct})

    return result


results = [None] * len(instances)

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(run_instance, inst, i): i for i, inst in enumerate(instances)}
    with tqdm.tqdm(total=len(instances)) as pbar:
        for future in as_completed(futures):
            r = future.result()
            results[r["idx"]] = r
            if is_ner:
                tqdm.tqdm.write(f"[row={r['row']}][F1={r['f1']:.2f}] Prediction: {r['pred'][:60]!r}  |  Expected: {r['gt'][:60]!r}")
            else:
                status = "✓" if r["correct"] else "✗"
                tqdm.tqdm.write(f"[row={r['row']}][{status}] Prediction: {r['pred'][:80]!r}  |  Expected: {r['gt_norm'][:80]!r}")
            pbar.update(1)

with open(SAVE_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    if is_ner:
        writer.writerow(["row", "task_id", "input", "GT", "raw_output", "parsed_output", "precision", "recall", "f1"])
        for r in results:
            writer.writerow([r["row"], r["task_id"], r["input"], r["gt"], r["raw"], r["pred"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])
    else:
        writer.writerow(["row", "task_id", "input", "GT", "raw_output", "parsed_output", "correct"])
        for r in results:
            writer.writerow([r["row"], r["task_id"], r["input"], r["gt_norm"], r["raw"], r["pred"], r["correct"]])

print(f"\n{'='*40}")
print(f"Task: {args.benchmark_task}")
if is_ner:
    avg_p  = sum(r["precision"] for r in results) / len(results)
    avg_r  = sum(r["recall"]    for r in results) / len(results)
    avg_f1 = sum(r["f1"]        for r in results) / len(results)
    print(f"Precision: {avg_p*100:.2f}%")
    print(f"Recall:    {avg_r*100:.2f}%")
    print(f"F1:        {avg_f1*100:.2f}%")
else:
    correct = sum(r["correct"] for r in results)
    print(f"Accuracy: {correct/len(results)*100:.2f}%  ({correct}/{len(results)})")
print(f"Saved to: {SAVE_PATH}")
print(f"{'='*40}")