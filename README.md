# Retinal Blood Vessel Segmentation
### IEEE Project — Digital Image Processing

---

## Paper Reference
**"An Efficient Retinal Blood Vessel Segmentation using Morphological Operations"**
Ozkava et al., IEEE ISMSIT 2018 | DOI: 10.1109/ISMSIT.2018.8567239

---

## Our Improvements Over the Paper

| # | Paper Does | We Do Instead |
|---|---|---|
| 1 | CLAHE directly on green channel | Top-Hat transform FIRST, then CLAHE |
| 2 | Only Otsu thresholding | Otsu + Adaptive merged (dual threshold) |
| 3 | Basic morphology only | Morphology + area-based false positive removal |
| 4 | Visual results only | Full metrics: Accuracy, Sensitivity, F1, etc. |
| 5 | DRIVE dataset only | DRIVE + STARE datasets |

---

## Setup

```bash
pip install opencv-python numpy matplotlib scipy scikit-image
```

---

## How to Run

### Option 1 — Demo (no dataset needed)
```bash
python demo_run.py
```
Generates output with a synthetic retinal image.

### Option 2 — With DRIVE Dataset (recommended)
1. Download DRIVE from: https://drive.grand-challenge.org
2. Extract images to `dataset/` folder
3. Run:
```bash
python segmentation.py dataset/01_test.tif dataset/01_manual1.gif
```

### Option 3 — Batch all DRIVE images
```bash
python batch_run.py dataset/ output/
```

---

## Output Files

| File | Description |
|---|---|
| `pipeline_visualization.png` | All preprocessing stages side by side |
| `method_comparison.png` | Paper method vs our method vs ground truth |
| `metrics_comparison_chart.png` | Bar chart of Accuracy/F1/Sensitivity |
| `results.json` | All metric values as JSON data |
| `*_segmented.png` | Final binary vessel mask |

---

## Project Structure

```
retinal_project/
├── code/
│   ├── segmentation.py   ← Main algorithm (paper + improvements)
│   ├── demo_run.py       ← Demo with synthetic image
│   └── README.md
├── dataset/              ← Put DRIVE images here
└── output/               ← Results saved here
```

---

## Member Task Division

| Member | Task | File Section |
|---|---|---|
| Member 1 | Preprocessing & Enhancement | `preprocess_pipeline()` |
| Member 2 | Segmentation & Morphology | `segment_pipeline()` |
| Member 3 | Evaluation & Results | `compute_metrics()` + charts |

---

## IEEE Citation Format

```
U. Ozkava, S. Ozturk, B. Akdemir and L. Sevfi, "An Efficient Retinal Blood
Vessel Segmentation using Morphological Operations," 2018 2nd International
Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT),
Ankara, Turkey, 2018, pp. 1-7, doi: 10.1109/ISMSIT.2018.8567239.
```
