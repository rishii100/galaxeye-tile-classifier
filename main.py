"""
main.py — FastAPI service for offline satellite tile classification.
Run with: .venv/bin/uvicorn main:app --reload
"""
import os
import json
import sqlite3
import hashlib
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ─── Config ───────────────────────────────────────────────────────────
SAFETENSORS_PATH = "model.safetensors"
CONFIG_PATH = "config.json"
MODEL_PATH = "model.pth"
DB_PATH = "results.db"
CONFIDENCE_THRESHOLD = 0.5

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Globals (set on startup) ────────────────────────────────────────
model = None
classes = None


# ─── Database helpers ─────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_id     TEXT UNIQUE NOT NULL,
            filename    TEXT NOT NULL,
            label       TEXT NOT NULL,
            confidence  REAL NOT NULL,
            probabilities TEXT,
            is_flagged  INTEGER DEFAULT 0,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            tile_hash   TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON classifications(label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_flagged ON classifications(is_flagged)")
    conn.commit()
    conn.close()


# ─── Model loading ────────────────────────────────────────────────────
def load_model():
    global model, classes
    
    # Check for Hugging Face Safetensors format first
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
        model = m
        print(f"✅ Loaded Hugging Face safetensors model ({len(classes)} classes): {classes}")
        return

    # Fallback to model.pth
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        classes = checkpoint["classes"]

        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, len(classes))
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        model = m
        print(f"✅ Loaded .pth model ({len(classes)} classes): {classes}")
        return

    raise RuntimeError(
        f"No model found! Expecting '{SAFETENSORS_PATH}' + '{CONFIG_PATH}' or '{MODEL_PATH}'."
    )


# ─── Inference ────────────────────────────────────────────────────────
def classify_image(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    confidence, pred_idx = probs.max(0)
    label = classes[pred_idx.item()]
    prob_dict = {cls: round(probs[i].item(), 4) for i, cls in enumerate(classes)}

    return label, confidence.item(), prob_dict


# ─── App lifecycle ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="GalaxEye Tile Classifier",
    description="Offline satellite tile land-use classification service",
    lifespan=lifespan,
)


def format_row(row):
    """Format an SQLite row dictionary with parsed JSON probabilities and boolean is_flagged."""
    item = dict(row)
    if isinstance(item.get("probabilities"), str):
        try:
            item["probabilities"] = json.loads(item["probabilities"])
        except Exception:
            pass
    if "is_flagged" in item:
        item["is_flagged"] = bool(item["is_flagged"])
    return item


# ─── Routes ───────────────────────────────────────────────────────────
@app.post("/classify")
async def classify_tile(file: UploadFile = File(...)):
    """Upload a satellite tile image → classify → store → return result."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    image_bytes = await file.read()

    # Deduplicate by hash
    tile_hash = hashlib.sha256(image_bytes).hexdigest()
    tile_id = os.path.splitext(file.filename)[0]

    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM classifications WHERE tile_id = ?", (tile_id,)
    ).fetchone()
    if existing:
        conn.close()
        return format_row(existing)

    # Classify
    label, confidence, prob_dict = classify_image(image_bytes)
    is_flagged = 1 if confidence < CONFIDENCE_THRESHOLD else 0

    # Store
    conn.execute(
        """INSERT INTO classifications
           (tile_id, filename, label, confidence, probabilities, is_flagged, tile_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tile_id, file.filename, label, round(confidence, 4),
         json.dumps(prob_dict), is_flagged, tile_hash),
    )
    conn.commit()
    conn.close()

    return {
        "tile_id": tile_id,
        "filename": file.filename,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "is_flagged": bool(is_flagged),
    }


@app.get("/results")
def get_results(
    label: Optional[str] = Query(None, description="Filter by predicted label"),
    flagged: Optional[bool] = Query(None, description="Filter flagged (low-confidence) predictions"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence threshold"),
    max_confidence: Optional[float] = Query(None, description="Maximum confidence threshold"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query stored classification results with optional filters."""
    conn = get_db()
    query = "SELECT * FROM classifications WHERE 1=1"
    params = []

    if label:
        query += " AND label = ?"
        params.append(label)
    if flagged is not None:
        query += " AND is_flagged = ?"
        params.append(int(flagged))
    if min_confidence is not None:
        query += " AND confidence >= ?"
        params.append(min_confidence)
    if max_confidence is not None:
        query += " AND confidence <= ?"
        params.append(max_confidence)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [format_row(r) for r in rows]


@app.get("/results/{tile_id}")
def get_result_by_tile(tile_id: str):
    """Get classification result for a specific tile."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM classifications WHERE tile_id = ?", (tile_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Tile '{tile_id}' not found.")
    return format_row(row)


@app.get("/stats")
def get_stats():
    """Aggregate stats: class distribution, avg confidence, flagged count."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM classifications").fetchone()["c"]
    if total == 0:
        conn.close()
        return {"total": 0, "message": "No classifications stored yet."}

    avg_conf = conn.execute("SELECT AVG(confidence) as a FROM classifications").fetchone()["a"]
    flagged = conn.execute("SELECT COUNT(*) as c FROM classifications WHERE is_flagged = 1").fetchone()["c"]

    dist = conn.execute("SELECT label, COUNT(*) as count FROM classifications GROUP BY label ORDER BY count DESC").fetchall()
    conn.close()

    return {
        "total_classified": total,
        "average_confidence": round(avg_conf, 4),
        "flagged_count": flagged,
        "class_distribution": {r["label"]: r["count"] for r in dist},
    }


@app.get("/health")
def health_check():
    """Basic health: model loaded, DB writable."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": classes,
        "db_exists": os.path.exists(DB_PATH),
    }
