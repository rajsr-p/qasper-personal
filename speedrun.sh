#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

output=$(uv run \
  --python 3.12 \
  --with rlms==0.1.1 \
  --with python-dotenv==1.2.2 \
  --with openai==2.26.0 \
  evals/rlm-script.py | tee /dev/stderr)

summary=$(echo "$output" | awk '/^Average over /,0')

{
  echo "# Eval Results"
  echo ""
  echo "**Run:** $(date '+%Y-%m-%d %H:%M:%S')"
  echo "**Model:** $(grep '^MODEL' evals/rlm-script.py | head -1 | cut -d'"' -f2)"
  echo ""
  echo '```'
  echo "$summary"
  echo '```'
} > EVAL.md

echo "→ EVAL.md written"
