from torchvision import datasets
from torch.utils.data import DataLoader
from src.transforms import get_transforms

train_transforms, val_test_transforms = get_transforms(image_size=224)

train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("data/val", transform=val_test_transforms)
test_dataset = datasets.ImageFolder("data/test", transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Class names:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
