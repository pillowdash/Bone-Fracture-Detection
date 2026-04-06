import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "outputs/models/best_model.pth"
IMAGE_PATH =  "data/test/normal/IMG0000375.jpg" #"data/test/fractured/IMG0000058.jpg" #"your_image.jpg"   # change this
CLASS_NAMES = ["fractured", "normal"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # same structure as training
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(image_path):
    device = get_device()

    model = build_model(num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    predicted_class = CLASS_NAMES[pred_idx]
    confidence = probs[0, pred_idx].item()

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {probs.cpu().numpy()}")


if __name__ == "__main__":
    predict_image(IMAGE_PATH)
