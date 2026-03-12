"""
Chunk QASPER papers via alphaxiv full-text into ~550-char sentence chunks.

Each entry: paper_id, title, abstract, paragraphs (chunks), question, evidence.
Only includes datapoints with at least one evidence piece.
"""

import json
import re
import urllib.request
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────
SPLIT = "test"
SAMPLE = False   # if True, output only the first valid datapoint
CHUNK_TARGET = 550

# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch_json(url):
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode())

def get_full_text(arxiv_id):
    """Fetch full text from alphaxiv. Returns a single string, or None on failure."""
    try:
        meta = fetch_json(f"https://api.alphaxiv.org/papers/v3/{arxiv_id}")
        version_id = meta["versionId"]
        data = fetch_json(f"https://api.alphaxiv.org/papers/v3/{version_id}/full-text")
        pages = data.get("pages", [])
        return "\n".join(p["text"] for p in pages)
    except Exception as e:
        print(f"  [skip] {arxiv_id}: {e}")
        return None

def make_chunks(text, target=CHUNK_TARGET):
    """Split text into ~target-length chunks at sentence boundaries.

    Splits at '. ' or '.\n' but not inside numbers like 4.2 (period
    preceded by a digit is left alone).
    """
    # re.split with capturing group returns [seg, sep, seg, sep, ..., seg]
    parts = re.split(r'(?<=[^0-9\s])\.([ \n])', text)
    units = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            units.append(parts[i] + '.' + parts[i + 1])
            i += 2
        else:
            units.append(parts[i])
            i += 1

    chunks = []
    current = ''
    for unit in units:
        current += unit
        if len(current) >= target:
            chunks.append(current.strip())
            current = ''
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading QASPER dataset...")
ds = load_dataset("allenai/qasper", trust_remote_code=True)
split_data = ds[SPLIT]

# ── Process ──────────────────────────────────────────────────────────────────
datapoints = []

total_papers = len(split_data)
for paper_idx, paper in enumerate(split_data):
    if paper_idx % 25 == 0:
        print(f"  Paper {paper_idx} / {total_papers}...")

    paper_id = paper["id"]

    full_text = get_full_text(paper_id)
    if full_text is None:
        continue

    chunks = make_chunks(full_text)

    qas = paper["qas"]
    for i, question in enumerate(qas["question"]):
        evidence = []
        for ans in qas["answers"][i]["answer"]:
            for ev in ans["evidence"]:
                if ev not in evidence:
                    evidence.append(ev)

        if not evidence:
            continue

        datapoints.append({
            "paper_id": paper_id,
            "title": paper["title"],
            "abstract": paper["abstract"],
            "paragraphs": chunks,
            "question": question,
            "evidence": evidence,
        })

        if SAMPLE:
            break

    if SAMPLE and datapoints:
        break

# ── Write ─────────────────────────────────────────────────────────────────────
suffix = "-sample" if SAMPLE else ""
out_path = f"chunk-qasper-{SPLIT}{suffix}.json"
with open(out_path, "w") as f:
    json.dump(datapoints, f, indent=2)

print(f"Wrote {len(datapoints)} datapoint(s) to {out_path}")
