import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import sys

# Add the model directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(os.path.dirname(current_dir), 'model')
sys.path.append(model_dir)

# Now we can import from train_model
from train_model import CNN

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
model.eval()

# Load class labels
with open(os.path.join(model_dir, "classes.json"), "r") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        return classes[class_index]
