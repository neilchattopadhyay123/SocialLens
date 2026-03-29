#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 YOUR_API_KEY [input_video_path] [output_video_path]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_BIN="${ROOT_DIR}/build/create_demo_video"
API_KEY="$1"
INPUT_VIDEO="${2:-${ROOT_DIR}/interview.mp4}"
OUTPUT_VIDEO="${3:-${ROOT_DIR}/attention_demo.mp4}"

if [[ ! -x "${BUILD_BIN}" ]]; then
  echo "Binary not found: ${BUILD_BIN}"
  echo "Run: ${ROOT_DIR}/scripts/build.sh"
  exit 1
fi

"${BUILD_BIN}" "${API_KEY}" "${INPUT_VIDEO}" "${OUTPUT_VIDEO}"
