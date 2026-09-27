# 🛰️ GalaxEye — Part 1: System Design Note
**Role:** Backend Engineer, ML Systems  
**Project:** Offline Satellite Tile Land-Use Classification Service  

---

## 1. System Architecture Overview

In an offline, isolated hardware environment (e.g., edge base stations, field labs, or compute modules aboard aerial/satellite ground units), the service operates under strict constraints: **zero internet access, bounded compute/memory, and no external daemon dependencies**.

The system is designed with a lightweight, decoupled 5-tier architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Offline Analyst Client                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST
┌───────────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Ingestion & Service Tier                  │
│   • Request Validation (MIME, 64x64 dimensions, payload bounds)        │
│   • Cryptographic Checksum Engine (SHA-256 for tile deduplication)     │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       Local CPU Inference Engine     │  │   Embedded Storage (SQLite)  │
│ • Zero-copy Safetensors Deserializer │  │ • Schema with WAL Mode       │
│ • ResNet-18 (ImageNet Pretrained)    │  │ • Top-1, Probabilities JSON  │
│ • Confidence Thresholding & Flagging │  │ • Multi-attribute Indexes   │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

### Components
1. **Ingestion Layer (FastAPI + Uvicorn)**: Asynchronous web service providing high-throughput endpoint handlers with schema validation via Pydantic and streaming file reads.
2. **Data Integrity & Deduplication (SHA-256)**: Computes byte-level SHA-256 hash on input tiles. Identical tiles bypass model inference and return cached classifications, preventing redundant compute.
3. **ML Inference Engine (PyTorch + Safetensors)**: Pretrained ResNet-18 fine-tuned on the 7 EuroSAT target classes. Deployed on CPU using Hugging Face `safetensors` format with zero-copy memory mapping (`mmap`).
4. **Offline Database (SQLite in WAL Mode)**: Embedded zero-dependency SQL database with indexed lookups on `label`, `confidence`, and `is_flagged`.
5. **Analyst Query Interface**: RESTful read endpoints supporting multi-predicate filtering, pagination, aggregate stats, and health diagnostics.

---

## 2. End-to-End Tile Flow

```
[Tile Upload: POST /classify]
          │
          ▼
   [1. Validate File] ──(Invalid)──► Return 400 Bad Request
          │ (Valid bytes)
          ▼
   [2. Compute SHA-256 Hash]
          │
          ├──(Hash/Tile ID Exists in DB)──► Return Stored Record (Dedup Cache Hit)
          │
          ▼ (New Tile)
   [3. Preprocessing]
          │  • PIL convert to RGB (standardize channel format)
          │  • Tensor normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
          ▼
   [4. CPU Inference (ResNet-18)]
          │  • Forward pass with torch.no_grad()
          │  • Softmax activation across 7 classes
          ▼
   [5. Confidence Evaluation]
          │  • confidence = max(probabilities)
          │  • is_flagged = 1 if confidence < 0.50 else 0
          ▼
   [6. Atomic Persistence (SQLite)]
          │  • Write tile_id, label, confidence, full JSON probabilities, is_flagged, tile_hash
          ▼
   [7. JSON Response to Client]
```

---

## 3. Real Decisions & Technical Trade-offs

### A. Handling Uncertain / Low-Confidence Predictions
* **Options Considered**:
  1. *Hard Drop / Error Return (HTTP 422)*: Rejecting uncertain tiles outright.
  2. *Top-1 Assignment without Qualification*: Storing the argmax class regardless of confidence.
  3. *Confidence Thresholding with Flagging (Selected)*: Storing the model prediction while marking `is_flagged = 1` when `confidence < 0.50`.
* **Rationale**: Satellite analysts cannot afford missing data. Rejecting an image creates geographic gaps in mapping, while blind acceptance introduces silent misclassification. Storing the full probability distribution and flagging uncertain tiles creates a **Human-in-the-Loop (HITL)** triage queue: analysts can prioritize reviewing only the flagged subset.

### B. What to Store (Data Schema Depth)
* **Options Considered**:
  1. *Minimalist*: Only `tile_id` and predicted `label`.
  2. *Comprehensive Audit (Selected)*: `tile_id`, `filename`, `label`, scalar `confidence`, full `probabilities` JSON map, `is_flagged` flag, `tile_hash`, and timestamp.
* **Rationale**: Model versions evolve. Storing the complete probability vector (`{"Forest": 0.48, "River": 0.42, ...}`) allows downstream analysts to re-evaluate decision boundaries or perform multi-label thresholding retroactively without re-running CPU inference over millions of historical tiles.

### C. Database Architecture: SQLite vs. PostgreSQL vs. DuckDB
* **Options Considered**:
  1. *PostgreSQL*: High concurrent write throughput, but requires running a background daemon, credential management, IPC sockets, and elevated memory overhead.
  2. *DuckDB*: Excellent columnar analytics, but less mature row-level write and concurrent lock semantics under streaming web server requests.
  3. *SQLite with WAL (Write-Ahead Logging) (Selected)*: Zero operational overhead, single-file portability, ACID compliance, and standard library inclusion. With WAL mode enabled (`PRAGMA journal_mode=WAL`), SQLite supports concurrent readers alongside a writer, easily handling local edge loads.

### D. Model Format: Legacy `.pth` (Pickle) vs. Hugging Face `safetensors`
* **Options Considered**:
  1. *PyTorch `.pth` Checkpoint*: Standard, but uses Python `pickle` (security vulnerability on isolated hardware if models are updated via physical drives) and requires memory reallocation on load.
  2. *Hugging Face `safetensors` + `config.json` (Selected)*: Pure tensor serialization with zero-copy memory mapping (`mmap`), instantaneous CPU cold-start, complete protection against arbitrary code execution, and native compatibility with Hugging Face Hub.

---

## 4. Assumptions & Questions for the GalaxEye Team

In designing this system, several assumptions were made based on typical edge satellite constraints. Below are the key operational ambiguities to clarify:

### Sensor & Ingestion Characteristics
1. **Sensor Radiometry & Bit Depth**: EuroSAT tiles are 8-bit RGB PNGs. Real satellite payloads (e.g., Sentinel-2, Landsat, SAR) produce 12-bit or 16-bit TIFFs with multi-spectral or synthetic aperture radar (SAR) bands (e.g., NIR, RedEdge, VV/VH). *Will the production service ingest raw GeoTIFFs or pre-processed orthorectified RGB chips?*
2. **Georeferencing & Spatial Indexing**: The current schema treats tiles as flat image entities. *Do tiles carry geospatial metadata (EPSG projection, bounding box coordinates, zoom level)?* If so, SQLite should be upgraded with SpatiaLite or R*Tree indexing to support spatial queries (`SELECT * WHERE ST_Intersects(...)`).

### Operational Environment & SLAs
3. **Ingestion Cadence**: *Is imagery received as a sporadic bulk dump (e.g., 50,000 tiles dumped upon satellite pass downlinks) or continuous low-volume streaming?* Bulk downlinks would favor an async Celery/Redis queue or background worker pool rather than synchronous HTTP requests.
4. **Hardware Footprint**: *What are the CPU core count, RAM ceilings, and storage drive write durability (e.g., eMMC vs NVMe) on the target deployed unit?* This determines whether ONNX Runtime or OpenVINO quantization (INT8) should be added to accelerate CPU inference.
