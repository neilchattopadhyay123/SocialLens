#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
ORT_ROOT_DEFAULT="${ROOT_DIR}/third_party/onnxruntime"

if [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
	ORT_ROOT="${ONNXRUNTIME_ROOT}"
elif [[ -f "${ORT_ROOT_DEFAULT}/include/onnxruntime_cxx_api.h" ]]; then
	ORT_ROOT="${ORT_ROOT_DEFAULT}"
elif [[ -f "/usr/local/onnxruntime/include/onnxruntime_cxx_api.h" ]]; then
	ORT_ROOT="/usr/local/onnxruntime"
else
	ORT_ROOT="${ORT_ROOT_DEFAULT}"
fi

echo "Using ONNX Runtime root: ${ORT_ROOT}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DONNXRUNTIME_ROOT="${ORT_ROOT}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Build complete: ${BUILD_DIR}/create_demo_video"
