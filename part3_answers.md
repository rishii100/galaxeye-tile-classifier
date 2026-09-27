# 🛰️ GalaxEye — Part 3: Problem Solving & Engineering Reasoning

---

### Question 1: Your classifier turns out to be wrong about 30% of the time (70% accuracy). What do you do — and how would you even decide whether it's "good enough" to be useful at all?

#### Part A: Deciding if it is "Good Enough"
Overall accuracy is a dangerous metric for satellite analytics. To determine operational utility:
1. **Asymmetric Cost Matrix**: The cost of a false positive vs. a false negative varies by land-use class. In environmental monitoring or disaster response, misclassifying a `Forest` or `River` as `Industrial` has severe consequences, whereas confusing `Residential` with `Industrial` at the urban boundary may carry minimal penalty. If the model achieves 95%+ precision on critical classes, a 70% aggregate accuracy may already provide immense operational value.
2. **Comparison with Baseline Workflows**: If human analysts manually triage 100% of incoming tiles, a classifier that flags the easiest 60% of tiles with 98% reliability and routes the ambiguous 40% to human experts reduces analyst workload by more than half. Utility is measured by **human time saved**, not model perfection.
3. **Information Entropy / Confidence Calibration**: If the model is wrong, *is it uncertain?* If wrong predictions cluster at low confidence scores ($< 0.50$), the system is well-calibrated and safe: thresholding easily isolates errors into the review queue.

#### Part B: Corrective Actions
1. **Confusion Matrix Decomposition**: Identify top confusion pairs (e.g., `AnnualCrop` vs `Forest`, or `Highway` vs `River`). For visually similar classes, introduce higher-resolution crops, multi-scale features, or color jitter augmentation.
2. **Confidence Threshold Tuning**: Calibrate decision thresholds per class rather than using a global 0.50 cutoff (e.g., raise threshold to 0.75 for `Highway` if precision is low).
3. **Data Quality & Label Audit**: Inspect misclassified candidate tiles. Small datasets (1,050 tiles) often suffer from ambiguous boundary crops or mislabeled ground truth.
4. **Hard Example Mining**: Capture low-confidence and misclassified tiles, add them to a fine-tuning partition, and re-train using focal loss or weighted cross-entropy.

---

### Question 2: This service runs offline, with no one watching it live. A month after deployment, how would you know it's still working correctly?

In an isolated environment with zero internet and no cloud APM (Datadog, Prometheus), observability must be built into the filesystem and local runtime:

1. **Canary / Golden Tile Heartbeat**:
   - Store 5 verified "canary tiles" on local disk.
   - Run a daily scheduled cron task (or startup self-test) feeding these canaries through the pipeline.
   - If output labels or confidence scores deviate from expected values by $> \epsilon$, log a hardware/model degradation alert in a local health status file (`/var/log/galaxeye_health.json`).
2. **Distribution Drift & Health Metrics**:
   - Record daily aggregate counters: total processed, average inference latency (ms), average confidence score, and class frequency distribution.
   - *Detection rule*: If the proportion of `SeaLake` spikes from 10% to 90%, or average confidence plummets from 0.88 to 0.40, the system has experienced sensor drift, focus failure, or lens contamination.
3. **Resource & Storage Telemetry**:
   - Log SQLite database size, write error counts, and filesystem disk space.
   - Enforce structured JSON logging with local log rotation (`logrotate` capping total log size at 500 MB) to prevent disk exhaustion.
4. **Offline Diagnostic Endpoint**:
   - Provide `GET /health` and `GET /stats` so an analyst plugging a laptop directly into the unit via local LAN/USB can immediately inspect runtime status, database integrity (`PRAGMA integrity_check`), and model uptime.

---

### Question 3: Tiles are coming in fine, but the stored results look wrong. Walk us through how you'd find the cause — your actual steps, in order.

When debugging a pipeline where tiles arrive but database entries look corrupted or nonsensical, I follow a strict **upstream-to-downstream isolation methodology**:

1. **Step 1: Check System & Application Logs**:
   - Inspect stdout/stderr logs for unhandled exceptions, dimension mismatch warnings, or SQLite lock retries.
2. **Step 2: Inspect Raw Ingested Bytes vs. Source**:
   - Extract a problematic stored `tile_hash` and compare it against the original incoming image file.
   - Confirm byte completeness: Did an incomplete transmission result in a partial or corrupt image?
3. **Step 3: Audit Image Decoding & Preprocessing Pipeline**:
   - Feed the stored image into the preprocessing script and dump the preprocessed array to disk.
   - *Check*: Was the color space altered (e.g., BGR vs. RGB swap in OpenCV/PIL)? Were normalization parameters applied twice or skipped? Were pixel values scaled to `[0, 1]` or left in `[0, 255]`?
4. **Step 4: Verify Class Mapping Alignment (`id2label`)**:
   - Verify whether the output index order of the model matches the class dictionary (`classes` array). A common bug occurs when `ImageFolder` classes are sorted alphabetically, but training code or model weights used a different label dictionary.
5. **Step 5: Run Direct Isolated Inference on Known Samples**:
   - Run inference on a canonical test image directly using Python REPL bypassing FastAPI.
   - If standalone inference is correct, the issue lies in request concurrency, async state mutation, or database insertion logic.
   - If standalone inference is wrong, the model weights or architecture definition were corrupted or mismatched.
6. **Step 6: Audit Database Insertion Logic**:
   - Inspect the parameterized `INSERT` query in SQLite. Check whether column ordering in the SQL statement matches the values tuple (e.g., swapping `confidence` with another column, or casting errors).

---

### Question 4: What's the weakest part of your design, and what would break it first?

#### The Weakest Part: Synchronous CPU Inference on the Request Thread

In the current working slice, tile preprocessing and ResNet-18 forward passes run **synchronously on the CPU during the HTTP POST request cycle**.

#### What Would Break It First: Bulk Tile Ingestion Spikes
1. **Failure Mode (HTTP Worker Starvation & Timeouts)**:
   - A single CPU forward pass takes ~25–50 ms.
   - If a batch of 1,000 tiles is sent concurrently over multiple client requests, all Uvicorn worker threads will saturate 100% of CPU cores immediately.
   - Subsequent HTTP requests (including health checks and analyst queries) will queue, experience latency spikes, and eventually drop with connection timeouts (`504 Gateway Timeout` or socket backlog overflow).
2. **Secondary Failure Mode (SQLite Write Contention)**:
   - SQLite operates with database-level locking during write operations. Under heavy concurrent writes from multiple threads, threads will encounter `sqlite3.OperationalError: database is locked`.

#### The Fix for Production:
Decouple ingestion from inference using a **Local Background Worker Architecture**:
- Ingestion endpoint validates the tile, saves the raw file to an incoming spool directory, writes a `QUEUED` status to SQLite, and immediately returns HTTP 202 Accepted (`{"tile_id": "...", "status": "queued"}`).
- A dedicated background worker process (with dynamic batching) pulls tiles in batches of 32 or 64, runs a single vectorized matrix multiplication on CPU, and commits results in bulk transactions (`executemany`), maximizing CPU cache utilization and eliminating HTTP thread starvation.
