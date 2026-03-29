#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
ORT_ROOT_DEFAULT="${ROOT_DIR}/third_party/onnxruntime"
ORT_ROOT="${ONNXRUNTIME_ROOT:-${ORT_ROOT_DEFAULT}}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DONNXRUNTIME_ROOT="${ORT_ROOT}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Build complete: ${BUILD_DIR}/create_demo_video"
