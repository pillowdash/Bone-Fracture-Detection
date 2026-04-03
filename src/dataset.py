from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader


def create_image_datasets(data_dir: str, train_transform, val_test_transform):
    """
    Create train, validation, and test ImageFolder datasets.

    Args:
        data_dir (str): Root data folder containing train/, val/, test/
        train_transform: torchvision transform for training data
        val_test_transform: torchvision transform for validation/test data

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)

    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 16,
    num_workers: int = 2
):
    """
    Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset
        batch_size (int): number of images per batch
        num_workers (int): number of worker processes

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def print_dataset_summary(train_dataset, val_dataset, test_dataset):
    """
    Print basic dataset summary.
    """
    print("Classes:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
