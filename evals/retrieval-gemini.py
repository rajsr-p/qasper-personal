import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
BATCH_SIZE = 10
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MODEL = "google/gemini-3-flash-preview"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """\
You are a precise information retrieval assistant. Given a question and a numbered list of paragraphs from a research paper, identify ALL paragraphs that contain evidence relevant to answering the question.

Rules:
- Include every paragraph that contains relevant evidence — do not omit any
- Do not include paragraphs that are irrelevant
- If no paragraph is relevant, return an empty list
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevant_indices",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indices of paragraphs that contain evidence relevant to the question",
                }
            },
            "required": ["indices"],
            "additionalProperties": False,
        },
    },
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def retrieve_relevant_indices(question: str, chunks: list[tuple[int, int, str]]) -> list[int]:
    numbered = "\n\n".join(f"[{i}] {c[2]}" for i, c in enumerate(chunks))
    user_msg = f"Question: {question}\n\nParagraphs:\n{numbered}"

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
    indices = result["indices"]
    return [i for i in indices if 0 <= i < len(chunks)]


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
    chunks = chunk_text(text)

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
        indices = retrieve_relevant_indices(question, chunks)
        retrieved_intervals = [(chunks[idx][0], chunks[idx][1]) for idx in indices]
    except Exception as e:
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})")
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
    print(f"  Precision : {total_precision / evaluated:.4f}")
    print(f"  Recall    : {total_recall / evaluated:.4f}")
    print(f"  F1        : {total_f1 / evaluated:.4f}")


if __name__ == "__main__":
    main()