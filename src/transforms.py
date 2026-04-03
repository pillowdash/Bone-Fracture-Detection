from torchvision import transforms


def get_transforms(image_size: int = 224):
    """
    Returns training and validation/test transforms for bone fracture detection.

    Args:
        image_size (int): Target image size for resizing. Default is 224.

    Returns:
        tuple: (train_transforms, val_test_transforms)
    """

    train_transforms = transforms.Compose([
        # Convert grayscale X-ray to 3 channels so it can work with ImageNet pretrained models
        transforms.Grayscale(num_output_channels=3),

        # Resize all images to the same input size expected by most pretrained models
        transforms.Resize((image_size, image_size)),

        # Randomly flip some images horizontally for augmentation
        transforms.RandomHorizontalFlip(p=0.5),

        # Slight rotation to simulate positioning variation
        transforms.RandomRotation(degrees=10),

        # Slight brightness and contrast adjustment for X-ray quality variation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),

        # Convert PIL image to PyTorch tensor
        transforms.ToTensor(),

        # Normalize using ImageNet mean/std because pretrained models were trained on ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transforms = transforms.Compose([
        # Convert grayscale X-ray to 3 channels
        transforms.Grayscale(num_output_channels=3),

        # Resize without augmentation
        transforms.Resize((image_size, image_size)),

        # Convert to tensor
        transforms.ToTensor(),

        # Normalize the same way as training images
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_test_transforms


# Optional: allow direct testing of this file
if __name__ == "__main__":
    train_tfms, val_tfms = get_transforms()
    print("Training transforms:")
    print(train_tfms)
    print("\nValidation/Test transforms:")
    print(val_tfms)
