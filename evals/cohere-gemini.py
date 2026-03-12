import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import cohere
from dotenv import load_dotenv

load_dotenv()

MAX_SAMPLES = None
BATCH_SIZE = 10
TOP_K = 10
RERANK_MODEL = "rerank-v4.0-pro"
MODEL = "google/gemini-3-flash-preview"

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

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


def retrieve_top_k(query: str, paragraphs: list[str], k: int = TOP_K) -> list[str]:
    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=paragraphs,
        top_n=k,
    )
    return [paragraphs[r.index] for r in response.results]


def retrieve_relevant_indices(question: str, paragraphs: list[str]) -> list[int]:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))
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
    return [i for i in indices if 0 <= i < len(paragraphs)]


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


def process_sample(i: int, sample: dict, total: int) -> dict | None:
    question = sample["question"]
    paragraphs = sample["paragraphs"]
    evidence = sample["evidence"]

    if not paragraphs or not evidence:
        print(f"[{i+1}/{total}] Skipping — no paragraphs or evidence")
        return None

    try:
        top_k = retrieve_top_k(question, paragraphs)
        indices = retrieve_relevant_indices(question, top_k)
        retrieved = [top_k[idx] for idx in indices]
    except Exception as e:
        print(f"[{i+1}/{total}] Skipping — retrieval failed ({e})")
        return None

    metrics = compute_metrics(retrieved, evidence)
    print(
        f"[{i+1}/{total}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} | retrieved={len(retrieved)} | {question[:60]}"
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
