import ast
import os
import json
import re
import ray
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

MAX_SAMPLES = None
SKIP = 0
BATCH_SIZE = 10
MODEL = "mercury-2"
USE_OPENROUTER = True
OPENROUTER_MODEL = "anthropic/claude-opus-4.6"

SYSTEM_PROMPT_TEMPLATE = """\
You are an evidence retrieval agent. The variable `TEXT` contains a research paper.

PAPER TITLE: {title}
ABSTRACT: {abstract}

QUESTION: {question}

{custom_tools_section}

RULES:
- CRITICAL: Each response you give must contain EXACTLY ONE ```repl block. Never two, never zero. \
You will be called multiple times. Each call = one block.
- You can only see the output of a block AFTER you submit it. \
So you CANNOT call extract_section() on the result of search() in the same response — you haven't seen the text yet.
- Do NOT call llm_query or rlm_query. Do NOT write string search code. Use ONLY the helpers.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of TEXT.
- The returned evidence must be self-contained: a reader seeing ONLY your returned text \
should have enough context to fully answer the question. Each evidence string MUST be at least \
one full paragraph (4+ sentences). Never return isolated sentences — always return the complete \
paragraph surrounding the key fact, including its topic sentence and any follow-up details.
- If you put two ```repl blocks in one response, the second block will be SILENTLY DROPPED. You will lose that work.
- Do NOT answer the question. Return the evidence substrings, nothing else.
- You can call search() multiple times in a single repl block to search for different keywords in parallel.
- If your initial search results lack promising snippets, search again with different \
query terms (synonyms, rephrased concepts, abbreviations). Don't repeat the same keywords.
- To expand a snippet, call search() on the snippet itself with a larger window \
and bidirectional=False. This re-finds the same location and returns more surrounding context. \
NOTE: you do not actually need to re-write out the snippet, it should be saved in an array/variable \
that you can just index. i.e. search(s[0], window=1000). When specifically trying to expand, we encourage \
window sizes of 1000+ characters.
- No narration, no explanation, no text outside code blocks.

search() prints every snippet with its starting character position. Read them carefully. \
After searching, identify ALL snippets that could be relevant — evidence is often spread across multiple sections \
of a paper (e.g. intro, methods, experiments may all contain relevant details). Expand each promising snippet generously. \
Then in the NEXT response (after you have read the expanded text), use extract_section to return \
the full section/paragraph that contains the evidence. Prefer returning too much over too little.

The procedure is exactly 3 responses:

Response 1:
```repl
s1 = search("keyword1")
s2 = search("keyword2")
```

Response 2 (after reading the search results, expand ALL promising snippets by re-searching them with a larger window):
```repl
e1 = search(s1[0], window=1000, bidirectional=False)
e2 = search(s1[3], window=1000, bidirectional=False)
e3 = search(s2[1], window=1000, bidirectional=False)
```

Response 3 (after reading the expanded text, return the full relevant paragraph(s) using extract_section. \
Always set start_phrase to a short phrase from the BEGINNING of the paragraph, not from the sentence containing the keyword, \
and end_phrase to a short phrase from the LAST sentence of the paragraph):
```repl
answer = [extract_section(e1[0], "beginning phrase of paragraph", "ending phrase of paragraph."), extract_section(e2[0], "beginning phrase of paragraph", "ending phrase of paragraph."), extract_section(e3[0], "beginning phrase of paragraph", "ending phrase of paragraph.")]
FINAL_VAR(answer)
```
"""



def _make_tools(text: str) -> dict:
    def _merge(items: list[tuple[int, str]]) -> list[tuple[int, str]]:
        if not items:
            return []
        intervals = sorted([(s, s + len(t)) for s, t in items])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [(s, text[s:e]) for s, e in merged]

    def search(keyword: str, window: int = 300, max_snippets: int = 10, bidirectional: bool = True) -> list[str]:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            if bidirectional:
                left = max(0, m.start() - window // 2)
                right = min(len(text), m.end() + window // 2)
            else:
                left = m.start()
                right = min(len(text), m.start() + window)
            while left > 0 and text[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > (window if bidirectional else 100):
                    break
            while right < len(text) and text[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(text) and text[right] in ".!?\n":
                right += 1
            results.append((left, text[left:right]))
        merged = _merge(results)
        shown = merged[:max_snippets]
        remaining = len(merged) - len(shown)
        snippets = []
        for start, snippet in shown:
            idx = len(snippets)
            print(f"--- snippet {idx} ---")
            print(snippet)
            snippets.append(snippet)
        if not shown:
            print(f"(no hits for {keyword!r})")
        if remaining > 0:
            print(f"(+{remaining} more)")
        return snippets

    def extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str:
        si = snippet.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        ei = snippet.lower().find(end_phrase.lower(), si)
        if ei == -1:
            result = snippet[si:]
        else:
            result = snippet[si:ei + len(end_phrase)]
        print(result)
        return result

    return {
        "search": search,
        "extract_section": extract_section,
        "TEXT": text,
    }


def retrieve_relevant_substrings(question: str, text: str, title: str = "", abstract: str = "") -> tuple[list[str], dict]:
    tools = _make_tools(text)
    system_prompt = (
        SYSTEM_PROMPT_TEMPLATE
        .replace("{title}", title)
        .replace("{abstract}", abstract)
        .replace("{question}", question)
    )
    if USE_OPENROUTER:
        backend = "openrouter"
        backend_kwargs = {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "model_name": OPENROUTER_MODEL,
        }
    else:
        backend = "openai"
        backend_kwargs = {
            "model_name": MODEL,
            "api_key": os.getenv("INCEPTION_API_KEY"),
            "base_url": "https://api.inceptionlabs.ai/v1",
        }

    rlm = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=1,
        max_iterations=20,
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
    title = sample.get("title", "")
    abstract = sample.get("abstract", "")
    paragraphs = sample["paragraphs"]
    evidence = sample["evidence"]

    print(f"[{i+1}/{total}] Question: {question}", flush=True)

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
        substrings, rlm_stats = retrieve_relevant_substrings(question, text, title, abstract)
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


@ray.remote
def process_sample_remote(i, sample, total):
    load_dotenv()
    return process_sample(i, sample, total)


def main():
    ray.init(ignore_reinit_error=True)

    with open("qasper-test.json") as f:
        data = json.load(f)

    samples = data[SKIP:] if MAX_SAMPLES is None else data[SKIP:SKIP + MAX_SAMPLES]

    total_precision = total_recall = total_f1 = 0.0
    total_time = total_input = total_output = total_cost = 0.0
    evaluated = 0

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start:batch_start + BATCH_SIZE]
        futures = [
            process_sample_remote.remote(batch_start + j, sample, len(samples))
            for j, sample in enumerate(batch)
        ]
        results = ray.get(futures)

        for result in results:
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

    ray.shutdown()

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


if __name__ == "__main__":
    main()
