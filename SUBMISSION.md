# HACKATHON SOLUTION SUBMISSION DOCUMENT

---

## TEAM DETAILS

| Field | Details |
|-------|---------|
| **Team Name** | FutureFormers |
| **Problem Statement Chosen** | Problem 2 |
| **Team Members** | Puranjay Gambhir (puranjay.gambhir@gmail.com) • Akshobhya Rao (akshobhyaraoap1845@gmail.com) • Rohan (snocky770@gmail.com) |
| **GitHub Repository Link** | https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0 |
| **Demo Link** | https://axes-map-checker.streamlit.app |

---

## 1. PROBLEM UNDERSTANDING & SCOPE

### 1.1 Explain the problem you are solving in your own words.

We are solving the problem of **detecting one specific type of error in road network data: endpoint gaps (broken route continuity)**.

When map vendors supply road network data as WKT LINESTRING geometries, a common and critical error occurs when two road segments that should connect have a **small coordinate mismatch** at their shared endpoint. For example, one segment ends at `(140, 220)` but the next segment starts at `(140.5, 220.3)` instead of exactly `(140, 220)`. This tiny gap breaks route continuity and causes routing failures.

- **Input:** A `.wkt` file containing LINESTRING geometries representing a street network.
- **Error Type:** ENDPOINT_GAP — dangling endpoints that are near but not touching another road segment.
- **Output:** A clear QA report listing every detected gap with its location, severity, confidence, and an auto-fix suggestion showing the exact coordinate correction needed.

### 1.2 What assumptions or simplifications did you make?

- **Single error type focus:** We detect only endpoint gaps (broken route continuity). Other error types like self-intersections, attribute errors, or topology violations are out of scope.
- **Coordinate system agnostic:** All coordinates in a single file share the same projected CRS. No CRS transformations are performed.
- **2D only:** Z-coordinates are ignored if present.
- **No manual labeling required:** All thresholds are derived from the dataset's own statistical distribution, making the system work on any dataset without tuning.
- **Adaptive thresholds:** The gap detection threshold uses a dual-signal approach: the 75th percentile of dangling-endpoint gaps combined with a scale-aware cap (15% of average segment length), preventing false positives on legitimate dead-end roads.

---

## 2. SOLUTION APPROACH & DESIGN

### 2.1 Describe your overall approach to solving the problem.

We built a **five-stage detection pipeline** focused entirely on endpoint gap detection, combining rule-based validation with unsupervised machine learning:

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: .wkt file                      │
│              (LINESTRING geometries)                    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE A: Feature Extraction (FeatureExtractor)          │
│  • Line length, vertex count, vertex density             │
│  • Start/end coordinates & endpoint degree               │
│  • Connectivity score (distance to nearest other road)   │
│  • Nearest-segment identification for each endpoint      │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE B: Gap Detection (GapDetector)                    │
│  • Find dangling endpoints (degree == 1)                 │
│  • Check if dangling endpoint is near another segment    │
│  • Adaptive threshold from dataset's own gap distribution│
│  • Confidence = 1 - (gap / threshold)                    │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE C: ML Anomaly Detection (AnomalyDetector)         │
│  • Isolation Forest (unsupervised) on feature matrix     │
│  • Flags segments with anomalous connectivity patterns   │
│  • Catches gaps that rule-based detection may miss       │
│  • Adjustable sensitivity parameter                      │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE D: Decision Logic (DecisionEngine)                │
│  • Merges rule-based + ML flags                          │
│  • Boosts confidence by 30% for double-flagged segments  │
│  • Assigns severity: HIGH (≥70%), MEDIUM (≥40%), LOW     │
│  • Deduplicates and sorts by confidence                  │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE E: Auto-Fix (AutoFixer)                           │
│  • Suggests snap coordinates for each detected gap       │
│  • Generates corrected .wkt file for download            │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  OUTPUT: Gap Report + Interactive Map + Corrected WKT    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Why did you choose this approach?

**Why endpoint gaps as the ONE error type?**

Endpoint gaps are the most common and impactful error in street network data:
- They directly break routing algorithms (a car can't travel through a gap)
- They're subtle — coordinates differ by only 0.1–1.0 units, invisible to the eye
- They're systematically fixable (snap the endpoint to the correct position)
- They directly align with the "no gaps" requirement for Option B

**Why the hybrid approach?**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pure rule-based (hardcoded thresholds) | Simple, fast | Breaks on different datasets/CRS | ❌ Rejected |
| Supervised ML (Random Forest / SVM) | High accuracy | Requires labeled training data we don't have | ❌ Rejected |
| Pure unsupervised ML | No labels needed | Hard to interpret, misses domain-specific errors | ❌ Rejected |
| **Hybrid: Data-driven rules + Isolation Forest** | **No labels, adaptive, interpretable** | Slightly more complex | ✅ **Chosen** |

---

## 3. TECHNICAL IMPLEMENTATION

### 3.1 Describe the technical implementation of your solution.

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Interactive single-page application |
| **Geometry Engine** | Shapely 2.0+ | WKT parsing, spatial computations, distance calculations |
| **ML Engine** | scikit-learn (Isolation Forest) | Unsupervised anomaly detection for connectivity patterns |
| **Data Processing** | Pandas, NumPy | Feature matrix, statistical analysis, threshold derivation |
| **Map Visualization** | Folium + streamlit-folium | Interactive map with severity-coded layers and popups |
| **Deployment** | Streamlit Cloud + GitHub | Auto-deploy on push to main branch |
| **UI Design** | Custom CSS (glassmorphism) | Professional interface with DM Sans typography |

**Architecture (OOP):**
- `FeatureExtractor` — builds endpoint map, computes per-segment connectivity features
- `GapDetector` — ONE rule: find dangling endpoints near but not touching other segments
- `AnomalyDetector` — StandardScaler + IsolationForest pipeline for connectivity anomalies
- `DecisionEngine` — static combiner with confidence boosting for double-flagged segments
- `AutoFixer` — computes snap coordinates and generates corrected WKT
- `build_error_report()` — generates the JSON export structure

### 3.2 What were the main technical challenges?

1. **Adaptive thresholds:** Every dataset has different coordinate scales. We derive ALL thresholds from the dataset itself — using the 75th percentile of dangling-endpoint gap distances, capped at 15% of average segment length to prevent dead-end roads from being flagged.

2. **No labeled training data:** Isolation Forest is unsupervised — it learns "normal" connectivity from the uploaded dataset and flags outliers automatically. No labels needed.

3. **Coordinate normalization for maps:** WKT data uses local projected coordinates, so Folium can't display them directly. We normalize relative to the centroid with a scale factor.

---

## 4. RESULTS & EFFECTIVENESS

### 4.1 What does your solution successfully achieve?

| Requirement | Status | Details |
|-------------|--------|---------|
| **Single error type** | ✅ Met | Focused entirely on ENDPOINT_GAP detection |
| **Clear output** | ✅ Met | Shows gap location, severity, confidence, and affected segments |
| **Training data / examples** | ✅ Met | 3 built-in error examples + 1 correct example with inline analysis |
| **Brief training instructions** | ✅ Met | Dedicated Training tab with step-by-step guide |
| **Demonstration on provided examples** | ✅ Met | Demo data (56 segs) + 3 crafted error examples with one-click analysis |
| **Auto-fix** | ✅ Met | Snap endpoints + download corrected WKT |
| **ML support** | ✅ Met | Isolation Forest catches anomalous connectivity patterns |
| **No hardcoded thresholds** | ✅ Met | All thresholds derived from data distribution |
| **Works with any .wkt file** | ✅ Met | CRS-agnostic, tested with multiple datasets |

### 4.2 How did you validate or test your solution?

We used **4 built-in example datasets** to demonstrate the detector:

| Example | Segments | Description | Expected | Actual |
|---------|----------|-------------|----------|--------|
| **Correct** | 5 | Fully connected network, exact endpoint matches | 0 gaps | ✅ 0 gaps |
| **Error 1** | 5 | Two gaps (~0.5 units) at segment junctions | 2+ gaps | ✅ Gaps detected |
| **Error 2** | 6 | Multiple gaps in a chain (0.2–0.4 units) | 3+ gaps | ✅ Multiple gaps |
| **Error 3** | 7 | Mix of near-miss (0.05) and larger (0.6) gaps | 3+ gaps | ✅ Graduated detection |

Additionally, the **provided hackathon dataset** (`Problem 2 - streets_xgen.wkt`, 56 segments) is loaded as demo data and correctly identifies connectivity issues.

---

## 5. INNOVATION & PRACTICAL VALUE

### 5.1 What is innovative or unique about your solution?

1. **Focused detection:** Instead of spreading thin across many error types, we go deep on ONE — endpoint gaps — with high accuracy and clear explanations.

2. **Hybrid rule + ML pipeline:** Rules catch domain-specific gaps; Isolation Forest catches unusual connectivity patterns. Double-flagged segments get a 30% confidence boost.

3. **Zero-configuration adaptive thresholds:** Derived from the dataset's own statistics. Works on any coordinate system without parameter tuning.

4. **Built-in training system:** The app includes a Training tab with step-by-step instructions for adding more examples, plus 3+1 built-in demonstration datasets that can be analyzed with one click.

5. **Auto-fix with download:** Not just detection — the system computes exact snap coordinates and lets users download a fully corrected `.wkt` file.

6. **Professional UI:** Glassmorphism design, interactive Folium maps with severity-coded markers, and exportable reports (CSV, JSON, TXT).

### 5.2 How can this be useful in production?

- **Map vendor QA:** Ride-hailing companies receiving road network updates can run this as an automated quality gate to catch broken connections before routing engines fail.
- **Batch processing:** The core classes (`FeatureExtractor`, `GapDetector`, `AnomalyDetector`) are UI-independent and can be imported as a Python library.
- **CI/CD integration:** The `error_report.json` output is machine-readable for automated acceptance/rejection pipelines.

---

## 6. LIMITATIONS & FUTURE IMPROVEMENTS

### 6.1 Current limitations

- **Single error type:** Only detects endpoint gaps, not other geometry issues.
- **No CRS awareness:** Cannot transform between coordinate systems.
- **Small dataset sensitivity:** Isolation Forest may be less reliable with < 10 segments.
- **2D only:** Z-coordinates are ignored.

### 6.2 Future improvements

1. **Add more error types** (one at a time): self-intersections, duplicate segments, unrealistically short links.
2. **CRS detection and transformation** using pyproj.
3. **Persistent feedback database** for supervised ML refinement over time.
4. **REST API / batch mode** via FastAPI for CI/CD integration.
5. **3D geometry support** for elevated road networks and bridges.

---

## TRAINING INSTRUCTIONS (SUMMARY)

The app includes a full **📚 Training** tab. Key points:

1. **Prepare WKT:** Create a `.wkt` file with LINESTRING geometries, one per line.
2. **Introduce errors:** Shift an endpoint by 0.1–1.0 units to create a gap.
3. **Upload & analyze:** The system automatically detects gaps with adaptive thresholds.
4. **Adjust sensitivity:** Use the ML Sensitivity slider in the sidebar (higher = more aggressive).
5. **Interpret results:** HIGH (≥70% confidence), MEDIUM (40-70%), LOW (<40%). Source: rule, ml, or rule+ml.
6. **Auto-fix:** Download a corrected `.wkt` with gaps snapped closed.

No manual labeling is ever needed — the Isolation Forest retrains on every new upload.

---

## FINAL DECLARATION

We confirm that this submission is our own work and was developed during the hackathon period.

| Field | Value |
|-------|-------|
| **Team Representative Name** | Puranjay Gambhir |
| **Team Members** | Puranjay Gambhir, Akshobhya Rao, Rohan |
| **Confirmation** | Yes |
| **Date** | February 2026 |
