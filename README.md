# 🛰️ GalaxEye — Offline Satellite Tile Land-Use Classification Service

This repository contains the complete solution for the offline satellite tile land-use classification service, designed to run on isolated hardware without internet connectivity.

---

## 📁 Submission Deliverables

| Deliverable | File | Description |
|---|---|---|
| **Part 1: Design Note** | [`design_note.md`](./design_note.md) | 1–2 page architecture specification, tile data flow, technical trade-offs, and questions for GalaxEye. |
| **Part 2: Working Slice** | [`main.py`](./main.py) | Runnable FastAPI service with CPU inference, SQLite persistence, and query endpoints. |
| **Part 3: Problem Solving** | [`part3_answers.md`](./part3_answers.md) | In-depth engineering answers covering 30% error mitigation, offline monitoring, debugging steps, and failure analysis. |
| **GPU Training Notebook** | [`train.ipynb`](./train.ipynb) | Google Colab / Kaggle T4 GPU notebook with automatic `.zip` extraction and Hugging Face `safetensors` export. |
| **Evaluation Script** | [`evaluate.py`](./evaluate.py) | Standalone local CPU script to evaluate model accuracy against the holdout set (`eval_set/`). |

---

## 🚀 Quickstart & Setup (Part 2)

The service runs locally on **Python 3.11** with **CPU only**.

### 1. Environment & Dataset Setup

First, ensure you have unzipped the provided dataset into the project root so that the `be-mlsys-assignment-dataset/` folder is present.

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Artifacts

The service loads the model via zero-copy memory mapping (`mmap`) using Hugging Face `safetensors`:
- `model.safetensors` (trained weights)
- `config.json` (class mappings & metadata)
- *(Fallback: `model.pth` is also supported automatically)*

> **Note:** If you train using [`train.ipynb`](./train.ipynb) on Google Colab (NVIDIA T4), the notebook exports `model.safetensors` and `config.json` ready to be placed in the project root.

### 3. Start the Offline Service

```bash
uvicorn main:app --reload --port 8000
```

Interactive Swagger documentation is available locally at:
👉 **`http://localhost:8000/docs`**

---

## 🧪 Testing the Endpoints (cURL Examples)

### A. Classify a Single Tile (Core Path: `POST /classify`)
Uploads a tile, validates the image, checks SHA-256 for duplicates, runs CPU ResNet-18 inference, flags predictions with confidence $< 0.50$, and stores the result in SQLite.

```bash
curl -X POST "http://localhost:8000/classify" \
     -F "file=@be-mlsys-assignment-dataset/eval_set/tile_001.png"
```

**Response Example:**
```json
{
  "tile_id": "tile_001",
  "filename": "tile_001.png",
  "label": "Forest",
  "confidence": 0.9186,
  "probabilities": {
    "AnnualCrop": 0.0,
    "Forest": 0.9186,
    "Highway": 0.0,
    "Industrial": 0.0,
    "Residential": 0.0,
    "River": 0.0813,
    "SeaLake": 0.0
  },
  "is_flagged": false
}
```

### B. Query Classified Results (`GET /results`)
Supports multi-predicate filtering and pagination for analysts:

```bash
# Get all flagged (uncertain) tiles
curl "http://localhost:8000/results?flagged=true"

# Filter by label with confidence threshold
curl "http://localhost:8000/results?label=Forest&min_confidence=0.80"
```

### C. Retrieve a Specific Tile (`GET /results/{tile_id}`)
```bash
curl "http://localhost:8000/results/tile_001"
```

### D. View Aggregate Service Statistics (`GET /stats`)
```bash
curl "http://localhost:8000/stats"
```

### E. Health & Diagnostics (`GET /health`)
```bash
curl "http://localhost:8000/health"
```

---

## 📊 Running Local Evaluation (Holdout Set)

To test model predictions on local CPU against the ground truth labels:

```bash
python evaluate.py
```

This runs all 210 tiles in `be-mlsys-assignment-dataset/eval_set/` against `eval_labels.csv` and outputs overall accuracy as well as per-class precision.

---

## 🏗️ Architecture & Decisions Summary

- **Offline-First**: Zero calls to external APIs or remote networks.
- **Deduplication Engine**: Uses SHA-256 hashing to prevent duplicate inference for previously seen tiles.
- **Safe Serialization**: Uses Hugging Face `safetensors` to prevent Python pickle injection vulnerabilities on isolated edge units.
- **Human-in-the-Loop Triage**: Low-confidence predictions (`< 0.50`) are marked with `is_flagged = 1` for analyst auditing.
- **Embedded Persistence**: SQLite with indexed fields ensures immediate read/write access with zero external database configuration.
