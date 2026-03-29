#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
ORT_DIR="${THIRD_PARTY_DIR}/onnxruntime"

ORT_VERSION="${1:-1.17.0}"
ARCHIVE_NAME="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE_NAME}"

mkdir -p "${THIRD_PARTY_DIR}"
cd "${THIRD_PARTY_DIR}"

if [[ -d "${ORT_DIR}" ]]; then
  echo "ONNX Runtime already present at ${ORT_DIR}"
  exit 0
fi

echo "Downloading ${DOWNLOAD_URL}"
curl -L -o "${ARCHIVE_NAME}" "${DOWNLOAD_URL}"

tar -xzf "${ARCHIVE_NAME}"
mv "onnxruntime-linux-x64-${ORT_VERSION}" "onnxruntime"

echo "ONNX Runtime installed at ${ORT_DIR}"
echo "Headers: ${ORT_DIR}/include"
echo "Library: ${ORT_DIR}/lib/libonnxruntime.so"
