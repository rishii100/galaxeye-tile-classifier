"""
train.py — Fine-tune ResNet-18 on candidate_tiles for land-use classification.
Saves model weights + class mapping to model.pth.
"""
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


DATASET_DIR = "be-mlsys-assignment-dataset"
CANDIDATE_DIR = os.path.join(DATASET_DIR, "candidate_tiles")
EVAL_DIR = os.path.join(DATASET_DIR, "eval_set")
EVAL_CSV = os.path.join(DATASET_DIR, "eval_labels.csv")
MODEL_PATH = "model.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train():
    print("🚀 Loading candidate_tiles...")
    train_dataset = datasets.ImageFolder(CANDIDATE_DIR, transform=TRANSFORM)
    classes = train_dataset.classes
    print(f"   Classes ({len(classes)}): {classes}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ResNet-18 with pretrained ImageNet weights, replace final FC for 7 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    epochs = 5
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"   Epoch {epoch+1}/{epochs}  loss={running_loss/total:.4f}  acc={correct/total*100:.1f}%")

    # Save checkpoint
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")

    return model, classes


def evaluate(model, classes):
    """Run eval_set through the model and compare against eval_labels.csv."""
    if not os.path.exists(EVAL_CSV):
        print("⚠️  eval_labels.csv not found, skipping evaluation.")
        return

    print("\n📊 Evaluating on eval_set...")
    model.eval()

    labels = {}
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = row["true_label"]

    correct, total = 0, 0
    with torch.no_grad():
        for filename, true_label in labels.items():
            img_path = os.path.join(EVAL_DIR, filename)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0)
            pred = classes[model(tensor).argmax(1).item()]
            correct += int(pred == true_label)
            total += 1

    print(f"🎯 Eval accuracy: {correct}/{total} ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    model, classes = train()
    evaluate(model, classes)
