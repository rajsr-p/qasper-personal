import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

MAX_SAMPLES = 1
BATCH_SIZE = 3  # RLM calls are heavier; keep concurrency low
MODEL = "google/gemini-3-flash-preview"

TASK_PROMPT = """\
You are given the full text of a research paper, split into paragraphs separated by "\\n\\n", \
and a question about it.

Your task: identify which paragraphs contain evidence relevant to answering QUESTION.

Use Python to split TEXT into paragraphs and collect the relevant ones into a list. \
Each element of the list must be an exact paragraph from TEXT (no paraphrasing or modifications).

When you have finished, store the list of relevant paragraphs in a variable named `answer` and \
call FINAL_VAR(answer). Example:
```python
answer = ["full paragraph text 1", "full paragraph text 2"]
FINAL_VAR(answer)
```

QUESTION: {question}

TEXT:
{text}
"""


def retrieve_relevant_substrings(question: str, text: str) -> list[str]:
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
    cost_str = f"${u.total_cost:.6f}" if u.total_cost is not None else "n/a"
    print(
        f"RLM STATS | time={result.execution_time:.2f}s "
        f"in={u.total_input_tokens:,} out={u.total_output_tokens:,} cost={cost_str}"
    )
    print("RLM RESPONSE", response)

    try:
        substrings = ast.literal_eval(response)
        if isinstance(substrings, list):
            return [s for s in substrings if isinstance(s, str)]
    except (ValueError, SyntaxError):
        pass

    return []


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
        print(f"[{i+1}/{total}] Skipping — no paragraphs or evidence")
        return None

    text = "\n\n".join(paragraphs)

    evidence_intervals = []
    for ev in evidence:
        ev = ev.strip()
        idx = text.find(ev)
        if idx == -1:
            print(f"[{i+1}/{total}] Warning — evidence not found in text: {ev[:60]}")
        else:
            evidence_intervals.append((idx, idx + len(ev)))

    if not evidence_intervals:
        print(f"[{i+1}/{total}] Skipping — no evidence located in text")
        return None

    try:
        substrings = retrieve_relevant_substrings(question, text)
        retrieved_intervals = []
        for s in substrings:
            idx = text.find(s)
            if idx != -1:
                retrieved_intervals.append((idx, idx + len(s)))
    except Exception as e:
        import traceback
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})")
        traceback.print_exc()
        return None

    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    print(
        f"[{i+1}/{total}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} | retrieved={len(retrieved_intervals)} | {question[:60]}"
    )
    return metrics


def main():
    with open("qasper-test.json") as f:
        data = json.load(f)

    samples = data if MAX_SAMPLES is None else data[:MAX_SAMPLES]

    total_precision = total_recall = total_f1 = 0.0
    evaluated = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {
            executor.submit(process_sample, i, sample, len(samples)): i
            for i, sample in enumerate(samples)
        }

        for future in as_completed(futures):
            metrics = future.result()
            if metrics is None:
                continue

            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_f1 += metrics["f1"]
            evaluated += 1

    print(f"\nAverage over {evaluated} samples:")
    if evaluated == 0:
        print("  No samples evaluated.")
        return
    print(f"  Precision : {total_precision / evaluated:.4f}")
    print(f"  Recall    : {total_recall / evaluated:.4f}")
    print(f"  F1        : {total_f1 / evaluated:.4f}")


if __name__ == "__main__":
    main()
