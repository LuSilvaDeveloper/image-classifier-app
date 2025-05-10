import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import sys
from train_model import CNN  # Make sure train_model.py defines the CNN class

# Load class labels
with open("classes.json", "r") as f:
    classes = json.load(f)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        class_label = classes[class_index]
        return class_label

# Run from CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)
    print(f"Predicted class: {result}")
