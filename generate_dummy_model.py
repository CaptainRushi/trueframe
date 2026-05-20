import torch
import torch.nn as nn
import os

class DummyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=224) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x).view(-1, 1)

class DummyReelsDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (batch, T, channels, H, W)
        self.conv = nn.Conv3d(3, 1, kernel_size=(10, 224, 224))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x is (batch, T, channels, H, W) -> transpose to (batch, channels, T, H, W)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.sigmoid(x).view(-1, 1)

os.makedirs("ai_service/models", exist_ok=True)

# 1. Generate Image model
model_path = "ai_service/models/efficientnet_b0_v1.onnx"
print("Generating dummy image ONNX model...")
model = DummyEfficientNet()
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, 
    dummy_input, 
    model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Dummy image model successfully saved to: {model_path}")

# 2. Generate Reels model
reels_model_path = "ai_service/models/trueframe_reels_detector.onnx"
print("Generating dummy reels ONNX model...")
reels_model = DummyReelsDetector()
reels_model.eval()
dummy_reels_input = torch.randn(1, 10, 3, 224, 224)

torch.onnx.export(
    reels_model, 
    dummy_reels_input, 
    reels_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Dummy reels model successfully saved to: {reels_model_path}")

