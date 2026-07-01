"""
Retinal Blood Vessel Segmentation using Morphological Operations
Implementation of: "An Efficient Retinal Blood Vessel Segmentation using Morphological Operations"
by Ozkaya et al., 2018

Pipeline:
  1. Extract Green Channel
  2. Adaptive Thresholding with 5x5 Gaussian Window
  3. Image Sharpening (Lab color space)
    4. Edge-Preserving Denoising
  5. Otsu Thresholding
  6. Morphological Opening
    7. Morphological Closing
    8. Circle Removing (Hough Transform)
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

# ─────────────────────────────────────────────
# STEP 1: Load image & extract Green Channel
# ─────────────────────────────────────────────
def extract_green_channel(image_bgr):
    """
    Extract green channel from BGR image.
    Green band has clearest blood vessel detail (least noise).
    Eq. 1: Gr(x,y) = I(x,y,2)
    """
    green = image_bgr[:, :, 1]   # OpenCV stores as BGR; index 1 = Green
    return green


# ─────────────────────────────────────────────
# STEP 2: Adaptive Thresholding (Gaussian 5×5)
# ─────────────────────────────────────────────
def adaptive_threshold_gaussian(green_img):
    """
    Adaptive threshold with 5×5 Gaussian window.
    Each pixel compared against weighted Gaussian mean of 5×5 neighbourhood.
    Eq. 2: G(x,y) = (1 / 2πσ²) * exp(-(x²+y²) / 2σ²)
    """
    # Block size must be odd; 5×5 window → blockSize=5
    # C is subtracted from weighted mean; tune for vessel visibility
    adaptive = cv2.adaptiveThreshold(
        green_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=5,
        C=2
    )
    return adaptive


# ─────────────────────────────────────────────
# STEP 3: Image Sharpening in Lab color space
# ─────────────────────────────────────────────
def sharpen_image(image_bgr):
    """
    Sharpen in Lab color space (L channel only).
    RGB → XYZ (CIE 1931) → Lab.
    3×3 sharpening kernel applied to L channel.
    Increases pixel difference to boost vessel/background contrast.
    """
    # Convert BGR → Lab
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # 3×3 sharpening filter (Figure 4 in paper)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)

    L_sharp = cv2.filter2D(L, ddepth=-1, kernel=kernel)
    L_sharp = np.clip(L_sharp, 0, 255).astype(np.uint8)

    # Merge and convert back to BGR
    lab_sharp = cv2.merge([L_sharp, a, b])
    sharpened_bgr = cv2.cvtColor(lab_sharp, cv2.COLOR_Lab2BGR)
    return sharpened_bgr


# ─────────────────────────────────────────────
# STEP 4: Edge-Preserving Denoising
# ─────────────────────────────────────────────
def apply_denoising(gray_img):
    """
    Edge-preserving denoising for vessel masks.
    Uses a small median pre-filter to remove isolated specks, then
    non-local means to smooth background noise while keeping vessel edges.
    """
    denoised = cv2.medianBlur(gray_img, 3)
    denoised = cv2.fastNlMeansDenoising(denoised, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return denoised


# ─────────────────────────────────────────────
# STEP 5: Otsu Thresholding
# ─────────────────────────────────────────────
def apply_otsu_threshold(gray_img):
    """
    Global Otsu thresholding to binarise softened image.
    Maximises inter-class variance between vessel (white) and background (black).
    Eq. 7–12 in the paper.
    """
    thresh_val = threshold_otsu(gray_img)
    binary = (gray_img >= thresh_val).astype(np.uint8) * 255
    return binary, thresh_val


# ─────────────────────────────────────────────
# STEP 6: Morphological Opening
# ─────────────────────────────────────────────
def morphological_opening(binary_img):
    """
    Remove small noise blobs that are smaller/more irregular than vessels.
    Opening = Erosion then Dilation  →  X ○ B = (X ⊖ B) ⊕ B  (Eq. 13)
    Structural element: 2×2 rectangle (keeps thin capillaries).
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)
    return opened


# ─────────────────────────────────────────────
# STEP 7: Morphological Closing
# ─────────────────────────────────────────────
def morphological_closing(binary_img):
    """
    Reconnect short breaks in vessel segments after opening.
    Closing = Dilation then Erosion.
    Uses a small ellipse so thin vessels are joined without heavy thickening.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)
    return closed


# ─────────────────────────────────────────────
# STEP 8: Circle Removal (Hough Transform)
# ─────────────────────────────────────────────
def remove_fov_circle(binary_img, original_bgr):
    """
    Fundus camera produces a circular FOV border artifact.
    Detect the circular boundary with Hough Circle Transform,
    then zero-out everything outside it.
    """
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = binary_img.shape
    min_r = int(min(h, w) * 0.35)
    max_r = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w),
        param1=50,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r
    )

    mask = np.zeros_like(binary_img)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        cx, cy, cr = circles[0]
        cv2.circle(mask, (cx, cy), cr - 5, 255, -1)
    else:
        # Fallback: use image centre with estimated radius
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 10, 255, -1)

    result = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    return result, mask


# ─────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────
def compute_metrics(segmented, ground_truth=None):
    """
    Compute Sensitivity, Specificity, Accuracy.
    If no ground truth provided, returns None values.
    """
    if ground_truth is None:
        return None, None, None

    seg_bin = (segmented > 0).astype(np.uint8)
    gt_bin  = (ground_truth > 0).astype(np.uint8)

    TP = np.sum((seg_bin == 1) & (gt_bin == 1))
    TN = np.sum((seg_bin == 0) & (gt_bin == 0))
    FP = np.sum((seg_bin == 1) & (gt_bin == 0))
    FN = np.sum((seg_bin == 0) & (gt_bin == 1))

    sensitivity = TP / (TP + FN + 1e-8) * 100
    specificity = TN / (TN + FP + 1e-8) * 100
    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-8) * 100

    return sensitivity, specificity, accuracy


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────
def segment_retinal_vessels(image_path, ground_truth_path=None, save_dir=None):
    """
    Run the complete segmentation pipeline and save intermediate results.
    Returns: final segmented image + metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    print(f"  Image size: {original.shape[1]}×{original.shape[0]} px")

    start_time = time.time()

    # --- Step 1: Green Channel ---
    print("  [1/8] Extracting green channel...")
    green = extract_green_channel(original)

    # --- Step 2: Adaptive Threshold ---
    print("  [2/8] Adaptive thresholding (5×5 Gaussian)...")
    adaptive = adaptive_threshold_gaussian(green)

    # --- Step 3: Image Sharpening ---
    print("  [3/8] Image sharpening (Lab color space)...")
    sharpened = sharpen_image(original)
    sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Combine adaptive threshold result with sharpened image
    combined = cv2.bitwise_and(sharpened_gray, adaptive)

    # --- Step 4: Denoising ---
    print("  [4/8] Edge-preserving denoising...")
    denoised = apply_denoising(combined)

    # --- Step 5: Otsu Threshold ---
    print("  [5/8] Otsu thresholding...")
    otsu_binary, otsu_thresh = apply_otsu_threshold(denoised)
    print(f"         Otsu threshold value: {otsu_thresh:.1f}")

    # --- Step 6: Morphological Opening ---
    print("  [6/8] Morphological opening...")
    morph_opened = morphological_opening(otsu_binary)

    # --- Step 7: Morphological Closing ---
    print("  [7/8] Morphological closing...")
    morph_closed = morphological_closing(morph_opened)

    # --- Step 8: Circle Removal ---
    print("  [8/8] Removing FOV circle artifact...")
    final_result, fov_mask = remove_fov_circle(morph_closed, original)

    elapsed = time.time() - start_time
    print(f"\n  ✓ Processing time: {elapsed:.3f} seconds")

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Compute metrics
    sens, spec, acc = compute_metrics(final_result, ground_truth)
    if sens is not None:
        print(f"  Sensitivity : {sens:.2f}%")
        print(f"  Specificity : {spec:.2f}%")
        print(f"  Accuracy    : {acc:.2f}%")

    metrics = {
        'time': elapsed,
        'sensitivity': sens,
        'specificity': spec,
        'accuracy': acc,
        'otsu_thresh': float(otsu_thresh)
    }

    # Build intermediate stages dict for plotting
    stages = {
        'original':       original,
        'green':          green,
        'adaptive':       adaptive,
        'sharpened_gray': sharpened_gray,
        'denoised':       denoised,
        'otsu':           otsu_binary,
        'morph':          morph_opened,
        'close':          morph_closed,
        'final':          final_result,
    }

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        for name, img in stages.items():
            out_path = os.path.join(save_dir, f"{base}_{name}.png")
            cv2.imwrite(out_path, img)
        print(f"\n  Intermediate images saved to: {save_dir}")

    return final_result, stages, metrics


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def plot_pipeline(stages, metrics, image_name="", save_path=None):
    """
    Create a clean multi-panel figure showing every pipeline stage.
    """
    fig = plt.figure(figsize=(21, 14))
    fig.patch.set_facecolor('#0d1117')

    titles = [
        ('original',       'Step 0\nOriginal RGB Image'),
        ('green',          'Step 1\nGreen Channel'),
        ('adaptive',       'Step 2\nAdaptive Thresholding\n(5×5 Gaussian)'),
        ('sharpened_gray', 'Step 3\nImage Sharpening\n(Lab color space)'),
        ('denoised',       'Step 4\nEdge-Preserving\nDenoising'),
        ('otsu',           'Step 5\nOtsu\nThresholding'),
        ('morph',          'Step 6\nMorphological\nOpening'),
        ('close',          'Step 7\nMorphological\nClosing'),
        ('final',          'Step 8\nFinal Result\n(Circle Removed)'),
    ]

    cols = 3
    rows = 3
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.04, right=0.96, top=0.88, bottom=0.05)

    for idx, (key, title) in enumerate(titles):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        img = stages[key]

        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cmap = 'gray' if key not in ['final', 'otsu', 'morph'] else 'Greys_r'
            ax.imshow(img, cmap=cmap)

        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.axis('off')

        # Highlight final result
        if key == 'final':
            for spine in ax.spines.values():
                spine.set_edgecolor('#00ff88')
                spine.set_linewidth(2)

    # Title & metrics
    met_str = ""
    if metrics['sensitivity'] is not None:
        met_str = (f"  Sensitivity: {metrics['sensitivity']:.2f}%  |  "
                   f"Specificity: {metrics['specificity']:.2f}%  |  "
                   f"Accuracy: {metrics['accuracy']:.2f}%  |  ")
    met_str += f"Process Time: {metrics['time']:.3f}s"

    fig.suptitle(
        f"Retinal Blood Vessel Segmentation Pipeline — {image_name}\n{met_str}",
        color='white', fontsize=11, fontweight='bold', y=0.97
    )

    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
        except OSError:
            root, ext = os.path.splitext(save_path)
            fallback_path = f"{root}_{int(time.time())}{ext}"
            fig.savefig(fallback_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            save_path = fallback_path
        print(f"  Figure saved: {save_path}")

    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────
# Batch Comparison Plots
# ─────────────────────────────────────────────
def plot_metrics_comparison(metrics_list, save_path=None):
    """
    Plot batch-level comparison graphs for timing and quality metrics.
    """
    if not metrics_list:
        return None

    images = [m['image'] for m in metrics_list]
    x = np.arange(len(images))

    times = [m['time'] for m in metrics_list]
    sens = [m['sensitivity'] if m['sensitivity'] is not None else np.nan for m in metrics_list]
    spec = [m['specificity'] if m['specificity'] is not None else np.nan for m in metrics_list]
    acc  = [m['accuracy'] if m['accuracy'] is not None else np.nan for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor('#0d1117')

    metric_panels = [
        (axes[0, 0], times, 'Processing Time (s)', '#4c78a8'),
        (axes[0, 1], sens, 'Sensitivity (%)', '#72b7b2'),
        (axes[1, 0], spec, 'Specificity (%)', '#f58518'),
        (axes[1, 1], acc,  'Accuracy (%)', '#54a24b'),
    ]

    for ax, values, title, color in metric_panels:
        ax.set_facecolor('#111827')
        bars = ax.bar(x, values, color=color, edgecolor='white', linewidth=0.6)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(images, rotation=45, ha='right', color='white', fontsize=8)
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis='y', linestyle='--', alpha=0.25)

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            label = f"{value:.2f}"
            offset = max(values) * 0.01 if len(values) else 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    label, ha='center', va='bottom', fontsize=8, color='white', rotation=0)

    fig.suptitle(
        'Batch Comparison of Retinal Segmentation Results',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        except OSError:
            root, ext = os.path.splitext(save_path)
            fallback_path = f"{root}_{int(time.time())}{ext}"
            fig.savefig(fallback_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            save_path = fallback_path
        print(f"  Comparison graph saved: {save_path}")

    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────
# Run on all images in a folder
# ─────────────────────────────────────────────
def run_batch(image_dir, results_dir, gt_dir=None):
    """
    Process every image in image_dir and collect metrics.
    """
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    all_metrics = []
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        base = os.path.splitext(fname)[0]

        gt_path = None
        if gt_dir:
            # Extract image number e.g. "01" from "01_test"
            img_num = base.split('_')[0]
            # Try common DRIVE mask naming patterns
            candidates = [
                os.path.join(gt_dir, f"{img_num}_manual1.gif"),
                os.path.join(gt_dir, f"{img_num}_manual1.png"),
                os.path.join(gt_dir, f"{img_num}_manual1.tif"),
                os.path.join(gt_dir, f"{img_num}_test_mask.gif"),
                os.path.join(gt_dir, base + ".gif"),
                os.path.join(gt_dir, base + ".png"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    gt_path = candidate
                    print(f"  Found GT: {os.path.basename(candidate)}")
                    break
            if not gt_path:
                print(f"  No GT found for {base} — checked: {[os.path.basename(c) for c in candidates]}")

        result, stages, metrics = segment_retinal_vessels(
            img_path, gt_path, save_dir=os.path.join(results_dir, 'intermediates')
        )

        fig_path = os.path.join(results_dir, f"{base}_pipeline.png")
        plot_pipeline(stages, metrics, image_name=base, save_path=fig_path)

        metrics['image'] = base
        all_metrics.append(metrics)

    return all_metrics


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_DIR   = "dataset/training/images"
    RESULTS_DIR = "results"
    GT_DIR      = "dataset/training/1st_manual"   # ground truth masks for training images

    metrics_list = run_batch(IMAGE_DIR, RESULTS_DIR, gt_dir=GT_DIR)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Image':<20} {'Time(s)':<10} {'Sens%':<10} {'Spec%':<10} {'Acc%':<10}")
    print("-"*60)
    for m in metrics_list:
        sens = f"{m['sensitivity']:.2f}" if m['sensitivity'] is not None else "N/A"
        spec = f"{m['specificity']:.2f}" if m['specificity'] is not None else "N/A"
        acc  = f"{m['accuracy']:.2f}"    if m['accuracy']    is not None else "N/A"
        print(f"{m['image']:<20} {m['time']:<10.3f} {sens:<10} {spec:<10} {acc:<10}")

    comparison_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plot_metrics_comparison(metrics_list, save_path=comparison_path)