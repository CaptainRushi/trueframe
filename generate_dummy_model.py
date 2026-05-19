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

os.makedirs("ai_service/models", exist_ok=True)
model_path = "ai_service/models/efficientnet_b0_v1.onnx"

print("Generating dummy ONNX model...")
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

print(f"Dummy model successfully saved to: {model_path}")
