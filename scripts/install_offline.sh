#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${ROOT_DIR}/offline_bundle/wheelhouse"
REQ_FILE="${ROOT_DIR}/requirements-pi.txt"

if [[ ! -d "${WHEELHOUSE}" ]]; then
  echo "wheelhouse not found: ${WHEELHOUSE}"
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "requirements file not found: ${REQ_FILE}"
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"

echo "Offline install finished."
echo "Run: streamlit run gui/app1.py"
