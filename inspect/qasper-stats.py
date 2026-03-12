"""
QASPER dataset — evidence selection statistics.
"""

from datasets import load_dataset

# ── Load dataset ────────────────────────────────────────────────────────────
print("Loading QASPER dataset...\n")
ds = load_dataset("allenai/qasper", trust_remote_code=True)

for split_name, split_data in ds.items():
    total_questions = 0
    questions_with_evidence = 0
    questions_with_only_text_evidence = 0

    for paper in split_data:
        qas = paper["qas"]
        for i, question in enumerate(qas["question"]):
            total_questions += 1
            # collect evidence across all annotators for this question
            has_any_evidence = False
            all_text_only = True
            for ans in qas["answers"][i]["answer"]:
                if ans["evidence"]:
                    has_any_evidence = True
                    for ev in ans["evidence"]:
                        if ev.startswith("FLOAT SELECTED"):
                            all_text_only = False
                            break
                if not all_text_only:
                    break

            if has_any_evidence:
                questions_with_evidence += 1
                if all_text_only:
                    questions_with_only_text_evidence += 1

    num_papers = len(split_data)
    print(f"=== {split_name} ===")
    print(f"  Papers                          : {num_papers:,}")
    print(f"  Total questions                 : {total_questions:,}")
    print(f"  Avg questions per paper          : {total_questions / num_papers:.1f}")
    print(f"  Questions with evidence          : {questions_with_evidence:,} ({100 * questions_with_evidence / total_questions:.1f}%)")
    print(f"  Questions with only text evidence: {questions_with_only_text_evidence:,} ({100 * questions_with_only_text_evidence / total_questions:.1f}%)")
    print()
