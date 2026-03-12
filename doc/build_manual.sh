#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TEX=pySimNMR_manual

if command -v latexmk >/dev/null 2>&1; then
  echo "Building with latexmk..."
  latexmk -pdf ${TEX}.tex
else
  echo "latexmk not found; falling back to pdflatex/bibtex sequence..."
  pdflatex -interaction=nonstopmode ${TEX}.tex || true
  bibtex ${TEX} || true
  pdflatex -interaction=nonstopmode ${TEX}.tex || true
  pdflatex -interaction=nonstopmode ${TEX}.tex || true
  echo "Manual build sequence completed."
fi

echo "Output: ${TEX}.pdf"
