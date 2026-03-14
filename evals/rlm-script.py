import ast
import os
import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

MAX_SAMPLES = None
MODEL = "inception/mercury-2"

SYSTEM_PROMPT_TEMPLATE = """\
You are an evidence retrieval agent. The variable `TEXT` contains a research paper.

QUESTION: {question}

{custom_tools_section}

RULES:
- Output ONLY ```repl code blocks. No narration, no explanation, no text outside code blocks.
- Do NOT call llm_query or rlm_query. Do NOT write string search code. Use ONLY the helpers.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of TEXT.
- Each returned substring should be 2-5 sentences of surrounding context, not just the exact \
words that answer the question. Include the full sentence and relevant neighbors.
- Do NOT answer the question. Return the evidence substrings, nothing else.

search_all prints every hit's full text automatically. Read them, then select/trim.

Example (2 iterations):

```repl
hits = search_all(["keyword1", "keyword2", "method_name"])
```

```repl
answer = [select(hits, 0), trim(hits[2], "start phrase", "end phrase."), select(hits, 4)]
FINAL_VAR(answer)
```
"""


@dataclass
class Hit:
    start: int
    text: str

    def __repr__(self):
        return f"[pos={self.start} len={len(self.text)}] {self.text!r}"


def _make_tools(text: str) -> dict:
    def search(keyword: str, window: int = 300) -> list:
        hits = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            left = max(0, m.start() - window // 2)
            right = min(len(text), m.end() + window // 2)
            while left > 0 and text[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > window:
                    break
            while right < len(text) and text[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(text) and text[right] in ".!?\n":
                right += 1
            hits.append(Hit(start=left, text=text[left:right]))
        return _merge_hits(hits)

    def _merge_hits(hits: list) -> list:
        if not hits:
            return []
        intervals = sorted([(h.start, h.start + len(h.text)) for h in hits])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [Hit(start=s, text=text[s:e]) for s, e in merged]

    def search_all(keywords: list, window: int = 300) -> list:
        all_hits = []
        for kw in keywords:
            all_hits.extend(search(kw, window))
        merged = _merge_hits(all_hits)
        for i, h in enumerate(merged):
            print(f"--- hit {i} ---")
            print(h.text)
        if not merged:
            print("(no hits)")
        return merged

    def select(hits: list, index: int) -> str:
        return hits[index].text

    def trim(hit, start_phrase: str, end_phrase: str) -> str:
        t = hit.text if isinstance(hit, Hit) else hit
        s = hit.start if isinstance(hit, Hit) else 0
        si = t.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        ei = t.lower().find(end_phrase.lower(), si)
        if ei == -1:
            return t[si:]
        return text[s + si : s + ei + len(end_phrase)]

    def expand(hit, chars: int = 200, left: int | None = None, right: int | None = None):
        l = left if left is not None else chars
        r = right if right is not None else chars
        new_start = max(0, hit.start - l)
        new_end = min(len(text), hit.start + len(hit.text) + r)
        return Hit(start=new_start, text=text[new_start:new_end])

    return {
        "search": search,
        "search_all": search_all,
        "select": select,
        "trim": trim,
        "expand": expand,
        "TEXT": text,
        "Hit": Hit,
    }


def retrieve_relevant_substrings(question: str, text: str) -> tuple[list[str], dict]:
    tools = _make_tools(text)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{question}", question)
    rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": MODEL,
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        },
        environment="local",
        max_depth=1,
        max_iterations=5,
        verbose=True,
        custom_tools=tools,
        custom_system_prompt=system_prompt,
    )

    import time as _time
    for _attempt in range(3):
        try:
            result = rlm.completion("Find evidence substrings in TEXT for the question above.")
            break
        except (ValueError, RuntimeError) as e:
            print(f"Retry {_attempt+1}/3 — {e}", flush=True)
            if _attempt < 2:
                _time.sleep(2 ** _attempt)
                continue
            raise
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
