#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${1:-1.17.0}"
INSTALL_DIR="${2:-/usr/local/onnxruntime}"
ARCHIVE_NAME="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE_NAME}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ -d "${INSTALL_DIR}" ]]; then
  echo "ONNX Runtime already present at ${INSTALL_DIR}"
  exit 0
fi

cd "${TMP_DIR}"

echo "Downloading ${DOWNLOAD_URL}"
curl -L -o "${ARCHIVE_NAME}" "${DOWNLOAD_URL}"

tar -xzf "${ARCHIVE_NAME}"

echo "Installing ONNX Runtime to ${INSTALL_DIR}"
sudo mkdir -p "$(dirname "${INSTALL_DIR}")"
sudo rm -rf "${INSTALL_DIR}"
sudo mv "onnxruntime-linux-x64-${ORT_VERSION}" "${INSTALL_DIR}"

echo "ONNX Runtime installed at ${INSTALL_DIR}"
echo "Headers: ${INSTALL_DIR}/include"
echo "Library: ${INSTALL_DIR}/lib/libonnxruntime.so"
