import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
SKIP = 0
BATCH_SIZE = 10
MODEL = "mercury-2"
USE_OPENROUTER = True
OPENROUTER_MODEL = "google/gemini-3-flash-preview"

if USE_OPENROUTER:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
else:
    client = OpenAI(
        base_url="https://api.inceptionlabs.ai/v1",
        api_key=os.getenv("INCEPTION_API_KEY"),
    )

SYSTEM_PROMPT = """\
You are an evidence retrieval assistant. Given a question and numbered lines from a research paper, identify ALL lines that contain evidence relevant to answering the question.

The returned lines must be sufficient for a reader — seeing ONLY those lines — to fully answer the question. \
If answering the question requires understanding context, definitions, methodology, or motivation, include those lines too.

Rules:
- Evidence for a question is often distributed across multiple sections of the paper \
(e.g. introduction, methods, experiments, discussion). Search broadly — do not stop at the first or most obvious match.
- When a line is relevant, include the ENTIRE paragraph or logical block it belongs to. \
Do not cherry-pick isolated sentences from the middle of a paragraph — always include from the topic sentence \
through the concluding sentence of that block. Paragraphs are separated by section headers or blank lines.
- When in doubt about whether a line is relevant, include it. \
Missing relevant evidence is a much bigger error than including a borderline line.
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


def split_into_sentences(text: str) -> list[tuple[int, int, str]]:
    boundaries = [m.end() for m in re.finditer(r'[.!?](?=\s|$)', text)]
    if not boundaries or boundaries[-1] < len(text):
        boundaries.append(len(text))

    sentences = []
    pos = 0
    for boundary in boundaries:
        chunk = text[pos:boundary]
        stripped = chunk.strip()
        if stripped:
            leading = len(chunk) - len(chunk.lstrip())
            char_start = pos + leading
            char_end = char_start + len(stripped)
            sentences.append((char_start, char_end, stripped))
        pos = boundary
    return sentences


def retrieve_relevant_lines(question: str, lines: list[tuple[int, int, str]], title: str = "", abstract: str = "") -> list[int]:
    numbered = "\n".join(f"{i}| {line[2]}" for i, line in enumerate(lines))
    # print(f"\n--- Numbered lines ---\n{numbered}\n")
    header = ""
    if title:
        header += f"Paper title: {title}\n"
    if abstract:
        header += f"Abstract: {abstract}\n"
    if header:
        header += "\n"
    user_msg = f"{header}Question: {question}\n\nLines:\n{numbered}"

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL if USE_OPENROUTER else MODEL,
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
    lines = split_into_sentences(text)

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

    title = sample.get("title", "")
    abstract = sample.get("abstract", "")

    try:
        selected = retrieve_relevant_lines(question, lines, title, abstract)
        retrieved_intervals = [(lines[idx][0], lines[idx][1]) for idx in selected]
    except Exception as e:
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})")
        return None

    # print(f"\n--- Output line numbers ---")
    # for idx in selected:
    #     print(f"  {idx}| {lines[idx][2]}")
    # print()

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
