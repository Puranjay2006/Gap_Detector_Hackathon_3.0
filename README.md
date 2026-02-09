# 🔍 Route Continuity Gap Detector

**AI-powered QA tool that detects endpoint gaps in road network data** — where road segments should connect but have small coordinate mismatches, breaking route continuity.

Built for **IIT Mandi Hackathon 3.0** | Team **FutureFormers** | Company: **Axes Systems GmbH** | Problem Statement 2, Option B

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

> **Live Demo:** [https://gapdetectorhackathon30-jhx7edzznqoz3mdrvfgkb2.streamlit.app/](https://gapdetectorhackathon30-jhx7edzznqoz3mdrvfgkb2.streamlit.app/)

---

## 👥 Team FutureFormers

| Member | Email |
|--------|-------|
| **Puranjay Gambhir** | puranjay.gambhir@gmail.com |
| **Akshobhya Rao** | akshobhyaraoap1845@gmail.com |
| **Rohan Kumar** | snocky770@gmail.com |

---

## 🚀 Features

- **Dual Detection Engine** — Rule-based gap detection + Isolation Forest ML anomaly detection
- **Adaptive Thresholds** — No hardcoded values; thresholds derived from each dataset's statistics
- **Interactive Map** — Folium-powered visualization with hover highlights, clickable gap markers with detailed popups
- **Auto-Fix Engine** — Calculates exact snap coordinates to close detected gaps
- **7-Tab Analysis Interface** — Map, Gap Report, Auto-Fix, Examples, Training, Statistics, How It Works
- **Export Options** — Download reports as CSV, JSON, or TXT; download corrected `.wkt` files
- **Interactive Tutorial** — Step-by-step onboarding guide with progress tracking
- **Professional UI** — DM Sans typography, glassmorphism design, smooth CSS animations

---

## ⚙️ How It Works

Our **5-stage AI pipeline** detects broken road connections automatically:

```
 ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌───────────────┐    ┌──────────────┐
 │  Stage A     │    │  Stage B          │    │  Stage C        │    │  Stage D       │    │  Stage E      │
 │  Parse WKT   │───▶│  Extract Features │───▶│  Rule-Based     │───▶│  ML Anomaly    │───▶│  Decision &   │
 │  Input       │    │  (connectivity)   │    │  Gap Detection  │    │  Detection     │    │  Auto-Fix     │
 └─────────────┘    └──────────────────┘    └────────────────┘    └───────────────┘    └──────────────┘
```

| Stage | Description |
|-------|-------------|
| **A — Parse WKT** | Reads LINESTRING geometries, extracts start/end coordinates |
| **B — Feature Extraction** | Endpoint degree, nearest-neighbor distance, segment length, vertex density |
| **C — Rule-Based Detection** | Dangling endpoints near another road but without exact coordinate match → gap. Threshold: `min(Q75 of dangling gaps, 15% of avg segment length)` |
| **D — ML Detection** | Isolation Forest learns "normal" connectivity patterns; flags anomalous segments |
| **E — Decision & Fix** | Merges results, 30% confidence boost for rule+ML overlap, generates snap-fix coordinates |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0.git
cd Gap_Detector_Hackathon_3.0

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud

1. Fork/clone this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select this repo → main file: `app.py`
4. Click **"Deploy!"** — live in ~30 seconds

---

## 📁 Project Structure

```
Gap_Detector_Hackathon_3.0/
├── app.py                          # Main Streamlit application (~1970 lines)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SUBMISSION.md                   # Hackathon submission document
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── demo_files/                     # Sample WKT files for testing
│   ├── clean_network.wkt           # Network with no errors
│   ├── endpoint_gaps.wkt           # Network with endpoint gaps
│   ├── errors_mixed.wkt            # Mixed error scenarios
│   ├── isolated_segments.wkt       # Isolated/disconnected segments
│   └── short_segments.wkt          # Short segment edge cases
└── Problem Statement 2/            # Original hackathon materials
    ├── Axes Systems - Masai Hackathon - Problem 2.pdf
    └── Problem 2 - streets_xgen.wkt
```

---

## 📂 Demo Datasets

We have provided several test datasets in the `demo_files/` directory to help you test the application immediately. Click the links below to view or download them:

| File Name | Description | Link |
|-----------|-------------|------|
| **clean_network.wkt** | A perfect network with no errors. Good for baseline testing. | [View / Download](https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0/blob/main/demo_files/clean_network.wkt) |
| **endpoint_gaps.wkt** | Contains deliberate endpoint gaps (broken connectivity). | [View / Download](https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0/blob/main/demo_files/endpoint_gaps.wkt) |
| **errors_mixed.wkt** | A complex scenario with mixed error types. | [View / Download](https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0/blob/main/demo_files/errors_mixed.wkt) |
| **isolated_segments.wkt** | Contains segments that are completely disconnected (islands). | [View / Download](https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0/blob/main/demo_files/isolated_segments.wkt) |
| **short_segments.wkt** | Contains extremely short segments that may be noise. | [View / Download](https://github.com/Puranjay2006/Gap_Detector_Hackathon_3.0/blob/main/demo_files/short_segments.wkt) |

---

## 📊 Input Format

The tool accepts `.wkt` or `.txt` files with WKT LINESTRING geometries:

```
LINESTRING(x1 y1, x2 y2, x3 y3, ...)
LINESTRING(x1 y1, x2 y2, ...)
```

Coordinates can be in any projected coordinate system. A built-in demo dataset of 56 street segments is included for instant testing.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit (≥1.31.0) |
| **Geometry Engine** | Shapely (≥2.0.4) |
| **Map Visualization** | Folium + streamlit-folium |
| **ML Model** | scikit-learn Isolation Forest |
| **Data Processing** | Pandas, NumPy |
| **Typography** | DM Sans (Google Fonts) |

---

## 🎯 Error Type Detected

**ENDPOINT GAP** — Where two road segments should share an exact endpoint coordinate but have a small mismatch (e.g., one ends at `(140, 220)` but the next starts at `(140.5, 220.3)`). This breaks route continuity and causes navigation/routing failures.

Each detected gap includes:
- **Severity** — HIGH / MEDIUM / LOW
- **Confidence** — Percentage based on detection source
- **Source** — Rule-based, ML, or both (rule+ML gets 30% confidence boost)
- **Auto-fix** — Exact snap coordinates to close the gap

---

## 📜 License

MIT License — Feel free to use and modify!

---

**Built with ❤️ by Team FutureFormers for IIT Mandi Hackathon 3.0 | Axes Systems GmbH | © 2026**
