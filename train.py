import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import numpy as np

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# DATA AUGMENTATION (OPEN-WORLD ROBUST)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATASETS
# =========================
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=val_transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# CLASS IMBALANCE FIX
# =========================
# Use the labels ImageFolder already indexed; iterating the dataset would load
# and augment every image just to read its label.
class_counts = Counter(train_dataset.targets)

print("Class Distribution:", class_counts)

weights = torch.tensor([
    1.0 / class_counts[0],
    1.0 / class_counts[1]
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================
num_epochs = 10  # Slightly more training for robustness

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print("Training Loss:", running_loss / len(train_loader))

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Validation Accuracy:", 100 * correct / total)

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "lung_model.pth")
print("Model saved successfully!")

# =========================
# TEST EVALUATION
# =========================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nTest Results:")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds))