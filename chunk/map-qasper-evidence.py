"""
Map QASPER evidence strings to the nearest paragraph in each entry's
paragraphs array using fuzzy string matching (rapidfuzz).

Input:  chunk-qasper-{SPLIT}.json  (output of chunk-qasper.py)
Output: mapped-qasper-{SPLIT}.json  — same structure, but evidence[]
        contains paragraphs copied verbatim from the paragraphs array.
"""

import json
from rapidfuzz import fuzz

# ── Constants ────────────────────────────────────────────────────────────────
SPLIT      = "test"
MAX        = None        # max number of datapoints to process (None = all)
SCORE_FUNC = fuzz.partial_ratio
THRESHOLD  = 60       # minimum score to accept a match (0-100)

# ── Load ─────────────────────────────────────────────────────────────────────
in_path = f"chunk-qasper-{SPLIT}.json"
print(f"Loading {in_path}...")
with open(in_path) as f:
    data = json.load(f)

subset = data[:MAX] if MAX is not None else data
print(f"Processing {len(subset)} / {len(data)} datapoint(s)...")

# ── Map ───────────────────────────────────────────────────────────────────────
def best_matching_paragraph(evidence_str, paragraphs):
    """Return the paragraph with the highest fuzzy score against evidence_str."""
    best_score = -1
    best_para  = None
    for para in paragraphs:
        score = SCORE_FUNC(evidence_str, para)
        if score > best_score:
            best_score = score
            best_para  = para
    return best_para, best_score

results = []
for idx, entry in enumerate(subset):
    paragraphs = entry["paragraphs"]
    mapped_evidence = []

    for ev in entry["evidence"]:
        scores = sorted(
            ((SCORE_FUNC(ev, para), para) for para in paragraphs),
            reverse=True
        )
        best_score, best_para = scores[0]
        print(f"\nevidence: {ev[:120]!r}")
        for rank, (score, para) in enumerate(scores[:5], 1):
            print(f"  #{rank} score={score:5.1f} | {para[:100]!r}")
        if best_score >= THRESHOLD and best_para not in mapped_evidence:
            mapped_evidence.append(best_para)

    result = dict(entry)
    result["evidence"] = mapped_evidence
    results.append(result)

# ── Write ─────────────────────────────────────────────────────────────────────
out_path = f"mapped-qasper-{SPLIT}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote {len(results)} datapoint(s) to {out_path}")
