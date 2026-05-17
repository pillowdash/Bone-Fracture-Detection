import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = "outputs/models/best_model.pth"

# If AUTO_FIND_EXAMPLES = False, this image will be used
IMAGE_PATH = "data/test/fractured/IMG0001934.jpg"

# If True, automatically search test set for useful examples
AUTO_FIND_EXAMPLES = True

TEST_FRACTURED_DIR = "data/test/fractured"
TEST_NORMAL_DIR = "data/test/normal"

OUTPUT_DIR = "outputs/plots/gradcam"
CLASS_NAMES = ["fractured", "normal"]


def get_device():
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).to("cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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


def load_model(model_path, device):
    model = build_model(num_classes=2)

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_true_label_from_path(image_path):
    path_str = str(image_path).lower().replace("\\", "/")
    if "/fractured/" in path_str:
        return "fractured"
    if "/normal/" in path_str:
        return "normal"
    return "unknown"


def prepare_image(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")

    rgb_resized = image.resize((image_size, image_size))
    rgb_array = np.array(rgb_resized).astype(np.float32) / 255.0

    transform = get_transform(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0)

    return image, rgb_array, input_tensor


def predict(model, input_tensor, device):
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, probs.cpu().numpy()


def generate_gradcam(model, input_tensor, rgb_array, device):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=input_tensor.to(device),
        targets=None
    )[0]

    cam_image = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True
    )

    return grayscale_cam, cam_image


def save_gradcam_figure(
    original_pil,
    cam_image,
    image_path,
    true_label,
    pred_class,
    confidence,
    probs,
    output_subdir
):
    out_dir = Path(OUTPUT_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    output_path = out_dir / f"{stem}_gradcam.png"

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_pil)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cam_image)
    ax2.set_title(
        f"Grad-CAM\n"
        f"True: {true_label} | Pred: {pred_class}\n"
        f"Confidence: {confidence:.4f}\n"
        f"P(fractured)={probs[0][0]:.4f}, P(normal)={probs[0][1]:.4f}"
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def run_gradcam_for_image(model, image_path, device, output_subdir="single"):
    image_path = str(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    true_label = get_true_label_from_path(image_path)
    original_pil, rgb_array, input_tensor = prepare_image(image_path, image_size=224)
    pred_idx, confidence, probs = predict(model, input_tensor, device)
    pred_class = CLASS_NAMES[pred_idx]

    print("\n" + "=" * 70)
    print(f"Image: {image_path}")
    print(f"True label from path: {true_label}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: fractured={probs[0][0]:.4f}, normal={probs[0][1]:.4f}")

    _, cam_image = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        rgb_array=rgb_array,
        device=device
    )

    output_path = save_gradcam_figure(
        original_pil=original_pil,
        cam_image=cam_image,
        image_path=image_path,
        true_label=true_label,
        pred_class=pred_class,
        confidence=confidence,
        probs=probs,
        output_subdir=output_subdir
    )

    print(f"Saved Grad-CAM image to: {output_path}")
    return {
        "image_path": image_path,
        "true_label": true_label,
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
        "output_path": str(output_path),
    }


def find_example(model, folder, expected_true_label, desired_pred_label, device):
    folder = Path(folder)
    image_paths = sorted(folder.glob("*.jpg"))

    for image_path in image_paths:
        _, _, input_tensor = prepare_image(str(image_path), image_size=224)
        pred_idx, confidence, probs = predict(model, input_tensor, device)
        pred_class = CLASS_NAMES[pred_idx]
        true_label = get_true_label_from_path(str(image_path))

        if true_label == expected_true_label and pred_class == desired_pred_label:
            return {
                "image_path": str(image_path),
                "true_label": true_label,
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": probs,
            }

    return None


def auto_find_and_save_examples(model, device):
    print("\nSearching for useful examples...")

    targets = [
        {
            "name": "correct_fractured",
            "folder": TEST_FRACTURED_DIR,
            "expected_true_label": "fractured",
            "desired_pred_label": "fractured",
        },
        {
            "name": "false_negative_fractured",
            "folder": TEST_FRACTURED_DIR,
            "expected_true_label": "fractured",
            "desired_pred_label": "normal",
        },
        {
            "name": "correct_normal",
            "folder": TEST_NORMAL_DIR,
            "expected_true_label": "normal",
            "desired_pred_label": "normal",
        },
    ]

    found_any = False

    for target in targets:
        result = find_example(
            model=model,
            folder=target["folder"],
            expected_true_label=target["expected_true_label"],
            desired_pred_label=target["desired_pred_label"],
            device=device,
        )

        if result is None:
            print(f"\nCould not find example for: {target['name']}")
            continue

        found_any = True
        run_gradcam_for_image(
            model=model,
            image_path=result["image_path"],
            device=device,
            output_subdir=target["name"]
        )

    if not found_any:
        print("\nNo matching examples found.")


def main():
    device = get_device()
    print(f"Running on device: {device}")

    model = load_model(MODEL_PATH, device)

    if AUTO_FIND_EXAMPLES:
        auto_find_and_save_examples(model, device)
    else:
        run_gradcam_for_image(
            model=model,
            image_path=IMAGE_PATH,
            device=device,
            output_subdir="single"
        )


if __name__ == "__main__":
    main()
