import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
MODEL_PATH = "models/pneumonia_resnet18_CE.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing pipeline (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

def load_model():
    """Load trained ResNet18 model with weights."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # NORMAL vs PNEUMONIA
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def predict_image(image_bytes: bytes):
    """Run inference on a single image."""
    model = load_model()

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, 1)

    classes = ["NORMAL", "PNEUMONIA"]
    return {
        "prediction": classes[pred_class.item()],
        "confidence": round(conf.item(), 4)
    }
