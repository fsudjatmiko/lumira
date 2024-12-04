import torch
from model_base import SimpleCNN

# Load the trained model
model = SimpleCNN(output_class=14)
checkpoint = torch.load('best_checkpoint.pth', map_location=torch.device('mps'))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Create a dummy input tensor with the same shape as your model's input
dummy_input = torch.randn(1, 3, 128, 128)

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Model has been converted to ONNX format and saved as 'model.onnx'")
