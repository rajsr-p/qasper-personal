import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = 10
SKIP = 0
BATCH_SIZE = 10
WORDS_PER_LINE = 15
# MODEL = "google/gemini-3-flash-preview"
MODEL = "anthropic/claude-sonnet-4.6"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """\
You are a precise information retrieval assistant. Given a question and numbered lines from a research paper, identify ALL lines that contain evidence relevant to answering the question.

Rules:
- Include every line that contains relevant evidence — do not omit any
- Do not include lines that are irrelevant
- If no line is relevant, return an empty list
- Return ranges as [start_line, end_line] pairs (inclusive)
- Merge adjacent or overlapping ranges where possible
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevant_lines",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ranges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "description": "Ranges of relevant line numbers as [start, end] pairs (inclusive)",
                }
            },
            "required": ["ranges"],
            "additionalProperties": False,
        },
    },
}


def split_into_lines(text: str, words_per_line: int = WORDS_PER_LINE) -> list[tuple[int, int, str]]:
    words_with_pos = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]
    lines = []
    for i in range(0, len(words_with_pos), words_per_line):
        group = words_with_pos[i:i + words_per_line]
        char_start = group[0][0]
        char_end = group[-1][1]
        line_text = " ".join(w for _, _, w in group)
        lines.append((char_start, char_end, line_text))
    return lines


def retrieve_relevant_lines(question: str, lines: list[tuple[int, int, str]]) -> list[int]:
    numbered = "\n".join(f"{i}: {line[2]}" for i, line in enumerate(lines))
    # print("numbered", numbered)
    user_msg = f"Question: {question}\n\nLines:\n{numbered}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    line_numbers = set()
    for r in result["ranges"]:
        if len(r) == 2:
            start, end = r
            for n in range(max(0, start), min(end + 1, len(lines))):
                line_numbers.add(n)
    return sorted(line_numbers)


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
    lines = split_into_lines(text)

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
        selected = retrieve_relevant_lines(question, lines)
        retrieved_intervals = [(lines[idx][0], lines[idx][1]) for idx in selected]
    except Exception as e:
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})")
        return None

    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    print(
        f"[{i+1}/{total}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} | lines={len(selected)}/{len(lines)} | {question[:60]}"
    )
    return metrics


def main():
    with open("qasper-test.json") as f:
        data = json.load(f)

    samples = data[SKIP:]
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    total_precision = total_recall = total_f1 = 0.0
    evaluated = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {
            executor.submit(process_sample, i + SKIP, sample, len(samples) + SKIP): i
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
    print(f"  Precision : {total_precision / evaluated:.4f}")
    print(f"  Recall    : {total_recall / evaluated:.4f}")
    print(f"  F1        : {total_f1 / evaluated:.4f}")


if __name__ == "__main__":
    main()
