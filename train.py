import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet101, ResNet101_Weights
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset

# Defining Hyperparams
CLASSES = 50
EPOCHS = 50
INITIAL_LR = 1e-4
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Loading resnet101 model with pretrained weights
# Establishing custom final fully connected layer
model = resnet101(weights=ResNet101_Weights.DEFAULT)

# Freeze backbone layers
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, CLASSES)
model = model.to(device)
print("Imported Model")


# Defining Transforms
# Data Augmentations included: Resizing, Horizontal flip, random rotation, color jitter
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Transforms for test set as well
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Dataset loading, done from folder
dataset = datasets.ImageFolder(
    root="/home/raghu/Desktop/DOAS-Raghu-Nicky-Emil/Dataset/Animals_with_Attributes2/JPEGImages",
    transform=None
)

# Defining dataset sizes
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Applying random splits
train_subset, val_subset, test_subset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

# Applying transforms onto the splits (correctly using wrapper)
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)

train_dataset = TransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, test_transform)
test_dataset = TransformSubset(test_subset, test_transform)

# Inspection
print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Class names:", dataset.classes)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Defining Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=INITIAL_LR)


# Training function
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# Evaluation function
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# Training loop
best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    print("-" * 40)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")


# Test Evaluation
model.load_state_dict(torch.load("best_model.pth"))

test_loss, test_acc = evaluate(model, test_loader)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

