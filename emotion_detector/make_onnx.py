from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "FER_static_ResNet50_AffectNet.pt"
OUTPUT_PATH = SCRIPT_DIR / "emo_affectnet_opencv.onnx"

def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Build model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("batch_norm", "bn").replace("i_downsample", "downsample")
        if "conv_layer_s2_same" in name or "fc1" in name or "fc2" in name:
            continue
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False,
    )

    # Some ONNX Runtime builds in this environment only support IR<=9.
    # Keep graph opset at 11 but down-level model IR metadata for compatibility.
    onnx_model = onnx.load(str(OUTPUT_PATH))
    if onnx_model.ir_version > 9:
        onnx_model.ir_version = 9
        onnx.save(onnx_model, str(OUTPUT_PATH))

    print(f"Success! OpenCV-friendly ONNX exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()