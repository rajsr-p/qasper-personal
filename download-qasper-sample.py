"""
Download a small slice of the original QASPER dataset and save as JSON
for inspection. Uses the same schema as the Hugging Face dataset.

Loads from the Parquet conversion (refs/convert/parquet) because the
dataset's legacy loading script is no longer supported by the datasets library.
"""

from datasets import load_dataset

SPLIT = "train"
NUM_EXAMPLES = 10
OUT_PATH = "qasper-original-sample.json"

# Parquet branch; avoids deprecated dataset loading script
REVISION = "refs/convert/parquet"

print(f"Loading {SPLIT}[:{NUM_EXAMPLES}] from allenai/qasper (Parquet)...")
ds = load_dataset(
    "allenai/qasper",
    split=f"{SPLIT}[:{NUM_EXAMPLES}]",
    revision=REVISION,
)
ds.to_json(OUT_PATH, indent=2)
print(f"Wrote {len(ds)} example(s) to {OUT_PATH}")
