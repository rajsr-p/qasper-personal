import json
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
TOP_K = 5
RERANK_MODEL = "rerank-v4.0-pro"

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def retrieve_top_k(query: str, paragraphs: list[str], k: int = TOP_K) -> list[str]:
    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=paragraphs,
        top_n=k,
    )
    return [paragraphs[r.index] for r in response.results]


def compute_metrics(retrieved: list[str], evidence: list[str]) -> dict:
    hits = 0
    for ev in evidence:
        for para in retrieved:
            if ev.strip() in para or para.strip() in ev:
                hits += 1
                break

    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(evidence) if evidence else 0.0
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

        try:
            retrieved = retrieve_top_k(question, paragraphs)
        except Exception as e:
            print(f"[{i+1}/{len(samples)}] Skipping — rerank failed ({e})")
            continue

        metrics = compute_metrics(retrieved, evidence)

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
