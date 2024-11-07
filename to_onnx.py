import torch
import torch.onnx
from torchvision import models

# Load your trained model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 14)  # 14 output classes
)
model.load_state_dict(torch.load("computer_part_model.pth"))
model.eval()  # Set the model to evaluation mode

# Dummy input tensor to match model input dimensions (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(
    model,                       # Model to export
    dummy_input,                 # Example input
    "computer_part_model.onnx",  # Filename for the output ONNX model
    opset_version=11,            # ONNX version
    input_names=["input"],       # Input layer name
    output_names=["output"],     # Output layer name
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allows for dynamic batch sizes
)

print("Model exported to computer_part_model.onnx")
