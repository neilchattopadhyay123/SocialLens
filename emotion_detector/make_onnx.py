import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

# 1. Create the skeleton
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

# 2. Load the "Knowledge" with name re-mapping
checkpoint = torch.load('/Users/kevin/SocialLens/emotion_detector/FER_static_ResNet50_AffectNet.pt', map_location='cpu')
new_state_dict = OrderedDict()

# This loop translates Elena's names into the Standard skeleton names
for k, v in checkpoint.items():
    name = k
    name = name.replace('batch_norm', 'bn')
    name = name.replace('i_downsample', 'downsample')
    # These specific layers don't exist in the standard skeleton 
    # but were likely extra layers in the custom training script
    if 'conv_layer_s2_same' in name or 'fc1' in name or 'fc2' in name:
        continue
    new_state_dict[name] = v

# 3. Force load the translated names
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# 4. Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "emo_affectnet.onnx", 
                  input_names=['input'], output_names=['output'],
                  opset_version=12)

print("✅ Success! Your 'emo_affectnet.onnx' file is ready.")