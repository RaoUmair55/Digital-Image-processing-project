# Retinal Blood Vessel Segmentation

This repository contains an implementation of a classical image-processing pipeline for retinal blood vessel segmentation described in "An Efficient Retinal Blood Vessel Segmentation using Morphological Operations" (Ozkaya et al., 2018). The main script is `retinal_segmentation.py` which runs an 8-step pipeline that processes fundus images, removes circular field-of-view artifacts, and computes simple quality metrics against ground-truth masks.

**Key Pipeline Steps**
- Extract green channel (best vessel contrast)
- Adaptive thresholding (5×5 Gaussian window)
- Image sharpening in Lab color space
- Edge-preserving denoising (median + NL means)
- Otsu global thresholding
- Morphological opening (remove small noise)
- Morphological closing (reconnect vessel segments)
- Circle detection & masking to remove FOV artifacts (Hough Transform)

**Files**
- `retinal_segmentation.py`: main implementation and utilities for batch processing, plotting, metrics.
- `requirements.txt`: Python dependencies for running the code.

**Requirements**
- Python 3.8+ recommended
- Tested with: OpenCV, NumPy, scikit-image, Matplotlib

Quick install (in project folder):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

**Usage**
1. Prepare dataset with structure (example):

```
dataset/
  training/
    images/          # input RGB fundus images (png/jpg/...) 
    1st_manual/      # ground-truth masks (optional; DRIVE naming conventions supported)
```

2. Adjust dataset paths at the bottom of `retinal_segmentation.py` or supply your own when calling functions programmatically.

3. Run full batch pipeline (saves intermediate images and comparison plots):

```bash
python retinal_segmentation.py
```

4. Run single image programmatically (example):

```python
from retinal_segmentation import segment_retinal_vessels
result, stages, metrics = segment_retinal_vessels('dataset/training/images/01_test.tif',
                                                'dataset/training/1st_manual/01_manual1.gif',
                                                save_dir='results/intermediates')
```

**Outputs**
- `results/intermediates/`: saved intermediate stage images for each processed input
- `results/<image>_pipeline.png`: multi-panel visualization of pipeline stages
- `results/metrics_comparison.png`: batch-level metric comparison plot (if ground-truth provided)

**Metrics**
The script computes Sensitivity, Specificity and Accuracy when ground-truth masks are available. These are printed to stdout and included in saved comparison plots.

**Customization & Tuning**
- Tweak `adaptive_threshold_gaussian()` block size and `C` value for different image qualities.
- Adjust denoising parameters (`fastNlMeansDenoising`) to trade-off noise suppression vs. thin-vessel preservation.
- Morphological structuring elements (size/shape) control how thin vessels are preserved vs. noise removal.
- The Hough circle parameters in `remove_fov_circle()` may require tuning for non-standard FOV sizes.

**Notes & Limitations**
- This is a classical image-processing pipeline (no learnable model). It works well on clean fundus datasets but may struggle on images with heavy artifacts, uneven illumination, or non-standard optics.
- Ground-truth compatibility: the batch loader tries common DRIVE dataset naming conventions; adapt `run_batch()` if you use a different dataset layout.

**Citation**
If you use this code in academic work, please cite the original method:

Ozkaya, U., et al., "An Efficient Retinal Blood Vessel Segmentation using Morphological Operations", 2018.

**License**
Add an appropriate license for your project (e.g., MIT). This repository is currently unlicensed; consider adding a `LICENSE` file.

**Contact / Next Steps**
- For problems running the script, ensure `requirements.txt` is installed and that your image paths exist.
- If you'd like, I can: update the script to accept CLI arguments, add a small test driver, or containerize the environment with Docker.

---
Generated for the project script `retinal_segmentation.py` by a helper script.