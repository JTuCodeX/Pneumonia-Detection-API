import io
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

MODEL_PATH = "models/pneumonia_resnet18_CE.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing: MUST match training pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def _build_model():
    """Build and load the model once."""
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Load model once (cached for all requests in this process)
MODEL = _build_model()
CLASSES = ["NORMAL", "PNEUMONIA"]

def predict_image(image_bytes: bytes):
    # 1) Open image and fix orientation if needed
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    # 2) Preprocess (resize, to tensor, normalize)
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # shape: (1, C, H, W)

    # 3) Inference
    with torch.no_grad():
        outputs = MODEL(img_tensor)            # logits
        probs = torch.softmax(outputs, dim=1) # probabilities

    # 4) Postprocess - build readable JSON
    prob_list = probs.squeeze(0).cpu().tolist()  # e.g. [0.03, 0.97]
    conf, pred_idx = torch.max(probs, 1)
    return {
        "prediction": CLASSES[pred_idx.item()],
        "confidence": round(conf.item(), 4),
        "probabilities": {CLASSES[i]: round(p, 4) for i, p in enumerate(prob_list)}
    }
