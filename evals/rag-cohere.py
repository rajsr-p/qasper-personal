import json
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
RERANK_MODEL = "rerank-v4.0-pro"

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    """Returns list of (start, end, chunk_text) tuples."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def retrieve_top_k(query: str, chunks: list[tuple[int, int, str]], k: int = TOP_K) -> list[tuple[int, int]]:
    """Returns (start, end) offsets of the top-k reranked chunks."""
    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[c[2] for c in chunks],
        top_n=k,
    )
    return [(chunks[r.index][0], chunks[r.index][1]) for r in response.results]


def union_size(intervals: list[tuple[int, int]]) -> int:
    """Total characters covered by a list of possibly-overlapping intervals."""
    return sum(e - s for s, e in merge_intervals(intervals))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection_size(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total characters in the intersection of two interval sets."""
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


def main():
    with open("qasper-test.json") as f:
        data = json.load(f)

    samples = data if MAX_SAMPLES is None else data[:MAX_SAMPLES]

    total_precision = total_recall = total_f1 = 0.0

    for i, sample in enumerate(samples):
        question = sample["question"]
        paragraphs = sample["paragraphs"]
        evidence = sample["evidence"]

        if not paragraphs or not evidence:
            print(f"[{i+1}/{len(samples)}] Skipping — no paragraphs or evidence")
            continue

        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text)

        # Locate each evidence string in the concatenated text
        evidence_intervals = []
        for ev in evidence:
            ev = ev.strip()
            idx = text.find(ev)
            if idx == -1:
                print(f"[{i+1}/{len(samples)}] Warning — evidence not found in text: {ev[:60]}")
            else:
                evidence_intervals.append((idx, idx + len(ev)))

        if not evidence_intervals:
            print(f"[{i+1}/{len(samples)}] Skipping — no evidence located in text")
            continue

        try:
            retrieved_intervals = retrieve_top_k(question, chunks)
        except Exception as e:
            print(f"[{i+1}/{len(samples)}] Skipping — rerank failed ({e})")
            continue

        metrics = compute_metrics(retrieved_intervals, evidence_intervals)

        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1"]

        print(f"[{i+1}/{len(samples)}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} | {question[:60]}")

    n = len(samples)
    print(f"\nAverage over {n} samples:")
    print(f"  Precision : {total_precision / n:.4f}")
    print(f"  Recall    : {total_recall / n:.4f}")
    print(f"  F1        : {total_f1 / n:.4f}")


if __name__ == "__main__":
    main()
