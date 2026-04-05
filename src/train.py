import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.transforms import get_transforms
from src.dataset import (
    create_image_datasets,
    create_dataloaders,
    print_dataset_summary,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace and unfreeze final classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def compute_metrics(all_labels, all_preds, fractured_index):
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, pos_label=fractured_index, zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds, pos_label=fractured_index, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds, pos_label=fractured_index, zero_division=0
    )
    return acc, precision, recall, f1


def train_one_epoch(model, dataloader, criterion, optimizer, device, fractured_index):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx % 20 == 0:
            print(f"  Training batch {batch_idx}/{len(dataloader)}")

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, precision, recall, f1 = compute_metrics(
        all_labels, all_preds, fractured_index
    )
    return epoch_loss, acc, precision, recall, f1


def evaluate(model, dataloader, criterion, device, fractured_index):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, precision, recall, f1 = compute_metrics(
        all_labels, all_preds, fractured_index
    )
    return epoch_loss, acc, precision, recall, f1


def main():
    # Config
    data_dir = "data"
    batch_size = 16
    num_workers = 2
    image_size = 224
    num_epochs = 10 # 5
    learning_rate = 1e-4
    freeze_backbone = False # True
    model_output_path = "outputs/models/best_model.pth"

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Transforms
    train_transforms, val_test_transforms = get_transforms(image_size=image_size)

    # Datasets
    train_dataset, val_dataset, test_dataset = create_image_datasets(
        data_dir=data_dir,
        train_transform=train_transforms,
        val_test_transform=val_test_transforms,
    )

    print_dataset_summary(train_dataset, val_dataset, test_dataset)

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Positive class
    fractured_index = train_dataset.class_to_idx["fractured"]
    print("Fractured class index:", fractured_index)

    # Model
    model = build_model(num_classes=2)
    model = model.to(device)

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)

    print("\nTrainable parameters:")
    for name in trainable_params:
        print(" ", name)

    # Class-weighted loss for imbalance
    targets = train_dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = class_counts.sum().float() / (
        len(class_counts) * class_counts.float()
    )
    class_weights = class_weights.to(device)

    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # Training loop
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, fractured_index
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device, fractured_index
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"Precision: {train_precision:.4f} | "
            f"Recall: {train_recall:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print("New best model found.")

    # Save best model
    model.load_state_dict(best_model_wts)
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"\nSaved best model to: {model_output_path}")

    # Test evaluation
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, criterion, device, fractured_index
    )

    print("\nFinal Test Results")
    print("=" * 50)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")


if __name__ == "__main__":
    main()
