import ast
import os
import json
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

MAX_SAMPLES = None
MODEL = "openai/gpt-5-mini"

TASK_PROMPT = """\
You are given the full text of a research paper and a question about it.

Your task: find all substrings from TEXT that contain evidence relevant to answering QUESTION.

Instructions:
- Use keyword/string search in Python to locate candidate excerpts efficiently. \
Think about what words or phrases would appear near the answer and search for them.
- NEVER pass the full TEXT or large chunks to llm_query — it is too slow and expensive. \
Any string passed to llm_query must be under 4000 characters.
- Only call llm_query on short candidate excerpts to judge their relevance.
- Returned substrings must be exact character-for-character slices of TEXT (use TEXT[start:end]). \
Do NOT paraphrase or modify them.
- Each substring should be full logical sentences / a paragraph-sized excerpt — not just a few \
words.

When you have finished, store the list of relevant substrings in a variable named `answer` and \
call FINAL_VAR(answer). Example:
```python
answer = ["exact slice 1", "exact slice 2"]
FINAL_VAR(answer)
```

QUESTION: {question}

TEXT:
{text}
"""


def retrieve_relevant_substrings(question: str, text: str) -> tuple[list[str], dict]:
    rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": MODEL,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        },
        environment="local",
        max_depth=2,
        max_iterations=10,
        verbose=False,
    )

    result = rlm.completion(TASK_PROMPT.format(question=question, text=text))
    response = result.response or ""

    u = result.usage_summary
    stats = {
        "time": result.execution_time,
        "input_tokens": u.total_input_tokens or 0,
        "output_tokens": u.total_output_tokens or 0,
        "cost": u.total_cost or 0.0,
    }
    cost_str = f"${stats['cost']:.6f}"
    print(
        f"RLM STATS | time={stats['time']:.2f}s "
        f"in={stats['input_tokens']:,} out={stats['output_tokens']:,} cost={cost_str}",
        flush=True,
    )
    print("RLM RESPONSE", response, flush=True)

    try:
        substrings = ast.literal_eval(response)
        if isinstance(substrings, list):
            return [s for s in substrings if isinstance(s, str)], stats
    except (ValueError, SyntaxError):
        pass

    return [], stats


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def union_size(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def intersection_size(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    a = merge_intervals(a)
    b = merge_intervals(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_metrics(retrieved_intervals: list[tuple[int, int]], evidence_intervals: list[tuple[int, int]]) -> dict:
    covered = intersection_size(retrieved_intervals, evidence_intervals)
    total_evidence = union_size(evidence_intervals)
    total_retrieved = union_size(retrieved_intervals)

    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def process_sample(i: int, sample: dict, total: int) -> dict | None:
    question = sample["question"]
    paragraphs = sample["paragraphs"]
    evidence = sample["evidence"]

    if not paragraphs or not evidence:
        print(f"[{i+1}/{total}] Skipping — no paragraphs or evidence", flush=True)
        return None

    text = "\n\n".join(paragraphs)

    evidence_intervals = []
    for ev in evidence:
        ev = ev.strip()
        idx = text.find(ev)
        if idx == -1:
            print(f"[{i+1}/{total}] Warning — evidence not found in text: {ev[:60]}", flush=True)
        else:
            evidence_intervals.append((idx, idx + len(ev)))

    if not evidence_intervals:
        print(f"[{i+1}/{total}] Skipping — no evidence located in text", flush=True)
        return None

    try:
        substrings, rlm_stats = retrieve_relevant_substrings(question, text)
        retrieved_intervals = []
        for s in substrings:
            idx = text.find(s)
            if idx != -1:
                retrieved_intervals.append((idx, idx + len(s)))
    except BaseException as e:
        import traceback
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})", flush=True)
        traceback.print_exc()
        return None

    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    print(
        f"[{i+1}/{total}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} | retrieved={len(retrieved_intervals)} | {question[:60]}",
        flush=True,
    )
    return {**metrics, "rlm_stats": rlm_stats}


def main():
    with open("qasper-test.json") as f:
        data = json.load(f)

    samples = data if MAX_SAMPLES is None else data[:MAX_SAMPLES]

    total_precision = total_recall = total_f1 = 0.0
    total_time = total_input = total_output = total_cost = 0.0
    evaluated = 0

    for i, sample in enumerate(samples):
        result = process_sample(i, sample, len(samples))
        if result is None:
            continue

        total_precision += result["precision"]
        total_recall += result["recall"]
        total_f1 += result["f1"]
        s = result["rlm_stats"]
        total_time += s["time"]
        total_input += s["input_tokens"]
        total_output += s["output_tokens"]
        total_cost += s["cost"]
        evaluated += 1

    print(f"\nAverage over {evaluated} samples:")
    if evaluated == 0:
        print("  No samples evaluated.")
        return
    print(f"  Precision : {total_precision / evaluated:.4f}")
    print(f"  Recall    : {total_recall / evaluated:.4f}")
    print(f"  F1        : {total_f1 / evaluated:.4f}")
    print(f"\nRLM stats (avg per sample):")
    print(f"  Time      : {total_time / evaluated:.2f}s")
    print(f"  Input tok : {total_input / evaluated:,.0f}")
    print(f"  Output tok: {total_output / evaluated:,.0f}")
    print(f"  Cost      : ${total_cost / evaluated:.6f}")
    print(f"  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
