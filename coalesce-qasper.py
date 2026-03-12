"""
Coalesce QASPER into a flat JSON array for evidence selection tasks.

Each entry: paper_id, paragraphs (flat list), question, evidence.
Only includes datapoints that have at least one evidence piece.
"""

import json
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────
SPLIT = "test"
SAMPLE = False   # if True, output only the first valid datapoint

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading QASPER dataset...")
ds = load_dataset("allenai/qasper", trust_remote_code=True)
split_data = ds[SPLIT]

# ── Coalesce ─────────────────────────────────────────────────────────────────
datapoints = []

for paper in split_data:
    paper_id = paper["id"]

    # Build flat paragraph list; prepend section title to the first paragraph
    # of each section.
    ft = paper["full_text"]
    paragraphs = []
    for sec_name, paras in zip(ft["section_name"], ft["paragraphs"]):
        for p_idx, para in enumerate(paras):
            text = f"{sec_name}: {para}" if (p_idx == 0 and sec_name) else para
            if text.strip():
                paragraphs.append(text)

    qas = paper["qas"]
    for i, question in enumerate(qas["question"]):
        # Collect unique evidence pieces across all annotators.
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
            "paragraphs": paragraphs,
            "question": question,
            "evidence": evidence,
        })

        if SAMPLE:
            break

    if SAMPLE and datapoints:
        break

# ── Write ─────────────────────────────────────────────────────────────────────
suffix = "-sample" if SAMPLE else ""
out_path = f"qasper-{SPLIT}{suffix}.json"
with open(out_path, "w") as f:
    json.dump(datapoints, f, indent=2)

print(f"Wrote {len(datapoints)} datapoint(s) to {out_path}")
