#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPORT_JOB="cifar10_traditional_cv_report"

cleanup_report_build() {
  rm -f "${REPORT_JOB}.aux" "${REPORT_JOB}.bbl" "${REPORT_JOB}.blg" "${REPORT_JOB}.log" "${REPORT_JOB}.out"
}
trap cleanup_report_build EXIT

xelatex -interaction=nonstopmode -halt-on-error -jobname="${REPORT_JOB}" main.tex
bibtex "${REPORT_JOB}"
xelatex -interaction=nonstopmode -halt-on-error -jobname="${REPORT_JOB}" main.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname="${REPORT_JOB}" main.tex

echo "Đã tạo report/${REPORT_JOB}.pdf"
