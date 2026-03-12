import json
import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
TOP_K = 5
EMBEDDING_MODEL = "google/gemini-embedding-001"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def embed(texts: list[str]) -> list[list[float]] | None:
    for attempt in range(2):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt == 0:
                print(f"  Embed failed ({e}), retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  Embed failed again ({e}), skipping.")
                return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query_emb: list[float], para_embs: list[list[float]], paragraphs: list[str], k: int = TOP_K) -> list[str]:
    scores = [cosine_similarity(query_emb, p) for p in para_embs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [paragraphs[i] for i in top_indices]


def compute_metrics(retrieved: list[str], evidence: list[str]) -> dict:
    # A retrieved paragraph is a hit if any evidence string is contained within it (or vice versa)
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

        # Embed query and all paragraphs together to save calls
        all_texts = [question] + paragraphs
        all_embs = embed(all_texts)
        if all_embs is None:
            print(f"[{i+1}/{len(samples)}] Skipping — embed failed")
            continue
        query_emb = all_embs[0]
        para_embs = all_embs[1:]

        retrieved = retrieve_top_k(query_emb, para_embs, paragraphs)
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
