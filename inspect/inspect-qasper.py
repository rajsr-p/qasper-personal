"""
Inspect the QASPER dataset — focused on evidence selection.

For each sample: paper title, question, ground-truth evidence, and answer.
"""

from datasets import load_dataset

# ── Load dataset ────────────────────────────────────────────────────────────
print("Loading QASPER dataset...\n")
ds = load_dataset("allenai/qasper", trust_remote_code=True)
train = ds["train"]

# ── Show samples ────────────────────────────────────────────────────────────
NUM_PAPERS = 5
MAX_QUESTIONS = 2

for paper in train.select(range(NUM_PAPERS)):
    print("=" * 70)
    print(f"PAPER: {paper['title']}  [{paper['id']}]")
    print("=" * 70)

    # ── full_text paragraphs metadata ──
    ft = paper["full_text"]
    num_sections = len(ft["section_name"])
    section_para_counts = [len(paras) for paras in ft["paragraphs"]]
    total_paras = sum(section_para_counts)
    total_chars = sum(len(p) for paras in ft["paragraphs"] for p in paras)
    print(f"\n  Paragraphs: {num_sections} sections, {total_paras} paragraphs, {total_chars:,} chars total")
    print(f"  Paragraphs per section: {section_para_counts}")
    for s_idx, (sec_name, paras) in enumerate(zip(ft["section_name"], ft["paragraphs"])):
        print(f"\n  Section {s_idx}: \"{sec_name}\" ({len(paras)} paragraphs)")
        for p_idx, para in enumerate(paras):
            print(f"    P[{s_idx}][{p_idx}] ({len(para):,} chars): {para}")

    qas = paper["qas"]
    for i, question in enumerate(qas["question"][:MAX_QUESTIONS]):
        print(f"\n  Q: {question}")

        for ans in qas["answers"][i]["answer"]:
            # ── answer ──
            if ans["unanswerable"]:
                answer_str = "[unanswerable]"
            elif ans["free_form_answer"]:
                answer_str = ans["free_form_answer"]
            elif ans["extractive_spans"]:
                answer_str = " | ".join(ans["extractive_spans"])
            elif ans["yes_no"] is not None:
                answer_str = "Yes" if ans["yes_no"] else "No"
            else:
                answer_str = "[empty]"
            print(f"  A: {answer_str}")

            # ── ground-truth evidence ──
            if ans["evidence"]:
                total_chars = sum(len(ev) for ev in ans["evidence"])
                print(f"  Evidence ({len(ans['evidence'])} pieces, {total_chars:,} chars):")
                for j, ev in enumerate(ans["evidence"], 1):
                    tag = "FIGURE/TABLE" if ev.startswith("FLOAT SELECTED") else "TEXT"
                    print(f"    [{j}] ({tag}) {ev}")
            else:
                print("  Evidence: [none]")

    print()
