# SocialLens

SmartSpectra-based video demo generator that overlays:
- Attention level (bar + label)
- Emotion classification (ONNX Runtime inference)

This README provides a full, reproducible setup from a clean machine.

## 1. Repository Layout

- `src/attention_demo/`: app entrypoint and core demo logic
- `include/attention_demo/`: app module headers
- `emotion_detector/`: emotion model artifacts and exporter script
- `scripts/`: helper scripts for setup/build/run
- `initial_test/`: earlier prototype code (not required for main demo build)

## 2. Prerequisites (Ubuntu)

Install build dependencies:

```bash
sudo apt update
sudo apt install -y \
	build-essential cmake pkg-config curl \
	libopencv-dev \
	libgoogle-glog-dev
```

Install Python packages needed only if you want to re-export the ONNX model:

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user torch torchvision onnx
```

Note: SmartSpectra SDK/dev packages must already be installed and discoverable by CMake.

## 3. Install ONNX Runtime (CPU)

Recommended path (local, no system-wide install):

```bash
cd /path/to/SocialLens
./scripts/setup_onnxruntime.sh
```

This installs ONNX Runtime under:

- `third_party/onnxruntime/include`
- `third_party/onnxruntime/lib`

You can also specify another version:

```bash
./scripts/setup_onnxruntime.sh 1.18.1
```

## 4. Build

```bash
cd /path/to/SocialLens
./scripts/build.sh
```

This configures CMake with:
- source: repository root
- build dir: `build/`
- `ONNXRUNTIME_ROOT` defaulting to `third_party/onnxruntime`

If your ONNX Runtime is installed somewhere else:

```bash
ONNXRUNTIME_ROOT=/usr/local/onnxruntime ./scripts/build.sh
```

## 5. Run the Demo

```bash
cd /path/to/SocialLens
./scripts/run_demo.sh YOUR_API_KEY [input_video_path] [output_video_path]
```

Examples:

```bash
./scripts/run_demo.sh YOUR_API_KEY
./scripts/run_demo.sh YOUR_API_KEY interview.mp4 attention_demo.mp4
```

Equivalent direct binary invocation:

```bash
./build/create_demo_video YOUR_API_KEY interview.mp4 attention_demo.mp4
```

## 6. Re-export Emotion ONNX Model (Optional)

Use this only if you need to regenerate the model:

```bash
cd /path/to/SocialLens/emotion_detector
python3 make_onnx.py
```

This produces:
- `emotion_detector/emo_affectnet_opencv.onnx`

The app prefers this file first, then falls back to other ONNX model names.

## 7. Troubleshooting

### CMake says ONNX Runtime not found

Set explicit root:

```bash
ONNXRUNTIME_ROOT=/path/to/onnxruntime ./scripts/build.sh
```

Required files are:
- `include/onnxruntime_cxx_api.h` (or nested ONNX Runtime header layout)
- `lib/libonnxruntime.so`

### Runtime fails to initialize emotion detector

The app will continue and render `Emotion: Unavailable` if ONNX model load fails.
Check:
- model path exists (`emotion_detector/emo_affectnet_opencv.onnx`)
- ONNX Runtime library is present
- model format is compatible with installed ONNX Runtime

### SmartSpectra package not found by CMake

Ensure SmartSpectra SDK is installed and CMake package config is available in your environment.

## 8. One-command Workflow

```bash
cd /path/to/SocialLens
./scripts/setup_onnxruntime.sh && ./scripts/build.sh && ./scripts/run_demo.sh YOUR_API_KEY interview.mp4 attention_demo.mp4
```