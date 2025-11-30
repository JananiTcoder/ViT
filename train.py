# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vit_model import ViT
from cnn_model import SimpleCNN
import json

device = "cpu"

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total


# Training loop with EARLY STOPPING
def train_model(model, name):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    MAX_EPOCHS = 30
    PATIENCE = 5  # Early stop patience

    best_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": []
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"{name} | Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f} | Test Acc={test_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # ---------------------------
        # EARLY STOPPING LOGIC
        # ---------------------------
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING at Epoch {epoch} (No improvement for {PATIENCE} epochs)\n")
                break

    # Save final model weights
    torch.save(model.state_dict(), f"{name}.pth")

    # Save training history JSON
    with open(f"training_history_{name}.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nSaved {name}.pth and training_history_{name}.json\n")


# Train CNN
cnn = SimpleCNN().to(device)
train_model(cnn, "cnn_cifar10")

# Train ViT
vit = ViT().to(device)
train_model(vit, "vit_cifar10")
