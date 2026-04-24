#!/usr/bin/env bash
# Rebuild report.ipynb from the source script and export it to PDF.
# This does NOT use xelatex/LaTeX — it uses Playwright (headless Chromium).
# Run from anywhere:  bash report/make.sh
#
# Requires a one-time setup:
#   uv sync --extra report
#   uv run playwright install chromium

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

unset VIRTUAL_ENV
export PATH="$HOME/.local/bin:$PATH"

echo "[report] building notebook ..."
uv run python _build_notebook.py

echo "[report] converting to PDF ..."
uv run jupyter nbconvert --to webpdf report.ipynb

echo "[report] done -> $HERE/report.pdf"
