"""
evaluate.py — Local CPU evaluation script for offline satellite tile classification.
Runs the evaluation set (eval_set/) against ground truth (eval_labels.csv)
using the exported model.safetensors (or model.pth fallback).
"""

import os
import csv
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

DATASET_DIR = "be-mlsys-assignment-dataset"
EVAL_DIR = os.path.join(DATASET_DIR, "eval_set")
EVAL_CSV = os.path.join(DATASET_DIR, "eval_labels.csv")

SAFETENSORS_PATH = "model.safetensors"
CONFIG_PATH = "config.json"
MODEL_PATH = "model.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    """Load model weights on CPU using safetensors or fallback .pth."""
    if HAS_SAFETENSORS and os.path.exists(SAFETENSORS_PATH) and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        if id2label:
            classes = [id2label[str(i)] for i in range(len(id2label))]
        else:
            classes = cfg.get("classes", [])

        state_dict = load_file(SAFETENSORS_PATH, device="cpu")
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, len(classes))
        m.load_state_dict(state_dict)
        m.eval()
        print(f" Loaded model from {SAFETENSORS_PATH} with {len(classes)} classes")
        return m, classes

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        classes = checkpoint["classes"]
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, len(classes))
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        print(f" Loaded model from {MODEL_PATH} with {len(classes)} classes")
        return m, classes

    raise FileNotFoundError(
        "No model artifact found! Please ensure 'model.safetensors' + 'config.json' "
        "(or 'model.pth') is in the current directory."
    )


def main():
    if not os.path.exists(EVAL_CSV) or not os.path.exists(EVAL_DIR):
        print(f" Evaluation directory or CSV not found in {DATASET_DIR}")
        return

    model, classes = load_model()

    ground_truth = {}
    with open(EVAL_CSV, mode="r") as f:
        for row in csv.DictReader(f):
            ground_truth[row["filename"]] = row["true_label"]

    total = 0
    correct = 0
    per_class = {c: {"correct": 0, "total": 0} for c in classes}

    print(f" Evaluating {len(ground_truth)} holdout tiles on CPU...")

    with torch.no_grad():
        for filename, true_label in ground_truth.items():
            img_path = os.path.join(EVAL_DIR, filename)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = probs.max(0)
            pred_label = classes[pred_idx.item()]

            is_match = (pred_label == true_label)
            correct += int(is_match)
            total += 1

            if true_label in per_class:
                per_class[true_label]["total"] += 1
                if is_match:
                    per_class[true_label]["correct"] += 1

    overall_acc = (correct / total * 100.0) if total else 0.0
    print("\n" + "=" * 50)
    print(f" Overall Accuracy: {correct}/{total} ({overall_acc:.2f}%)")
    print("=" * 50)
    print("Per-class performance:")
    for c, stats in per_class.items():
        tot = stats["total"]
        cor = stats["correct"]
        acc = (cor / tot * 100.0) if tot else 0.0
        print(f"  • {c:<12}: {cor:2d}/{tot:2d} ({acc:5.1f}%)")


if __name__ == "__main__":
    main()
