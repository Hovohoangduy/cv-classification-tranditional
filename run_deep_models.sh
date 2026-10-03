#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# Chạy 200 epoch trên CIFAR-10; tham số CLI phía sau có thể ghi đè khi cần.
DEEP_OUTPUT_DIR=outputs/deep_models_200_gn
for argument in "$@"; do
  if [[ "$argument" == "--quick" ]]; then
    DEEP_OUTPUT_DIR=outputs/deep_models_quick
  fi
done
python -m deep_models.run --epochs 200 --output-dir "$DEEP_OUTPUT_DIR" "$@"
bash report/build.sh
