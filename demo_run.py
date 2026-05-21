"""
demo_run.py — Run segmentation on a synthetic retinal image
Use this to test your code BEFORE downloading DRIVE dataset.
Once you have DRIVE, use: python segmentation.py <image> <gt>
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation import (
    preprocess_pipeline, segment_pipeline,
    segment_pipeline_paper_only, compute_metrics,
    save_result_figure, save_comparison_figure,
    print_metrics_table, load_image
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def create_realistic_demo():
    """
    Synthetic retinal image that mimics DRIVE dataset structure.
    Vessels of varying width + uneven illumination + noise.
    """
    h, w = 565, 565
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient (simulates uneven retinal illumination)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
            val = int(max(0, 60 - dist * 0.08))
            img[y, x] = [val//3, val, val//3]

    # --- Draw blood vessels (tree structure) ---
    def draw_vessel(img, p1, p2, thickness, color=(100, 190, 100)):
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    cx, cy = 280, 280  # optic disc center

    # Main arteries (thick)
    draw_vessel(img, (cx, cy), (cx+180, cy-120), 5, (110, 200, 110))
    draw_vessel(img, (cx, cy), (cx-160, cy-100), 5, (110, 200, 110))
    draw_vessel(img, (cx, cy), (cx+150, cy+140), 4, (105, 195, 105))
    draw_vessel(img, (cx, cy), (cx-140, cy+130), 4, (105, 195, 105))

    # Secondary branches (medium)
    draw_vessel(img, (cx+180, cy-120), (cx+280, cy-60),  3, (95, 175, 95))
    draw_vessel(img, (cx+180, cy-120), (cx+240, cy-200), 3, (95, 175, 95))
    draw_vessel(img, (cx-160, cy-100), (cx-260, cy-50),  3, (95, 175, 95))
    draw_vessel(img, (cx-160, cy-100), (cx-200, cy-200), 3, (95, 175, 95))
    draw_vessel(img, (cx+150, cy+140), (cx+270, cy+80),  2, (90, 170, 90))
    draw_vessel(img, (cx+150, cy+140), (cx+210, cy+240), 2, (90, 170, 90))

    # Thin capillaries (thin, low contrast — what the paper misses!)
    draw_vessel(img, (cx+280, cy-60),  (cx+340, cy-10),  1, (80, 155, 80))
    draw_vessel(img, (cx+240, cy-200), (cx+310, cy-250), 1, (80, 155, 80))
    draw_vessel(img, (cx-260, cy-50),  (cx-330, cy-10),  1, (80, 155, 80))
    draw_vessel(img, (cx-200, cy-200), (cx-270, cy-250), 1, (80, 155, 80))
    draw_vessel(img, (cx+270, cy+80),  (cx+340, cy+120), 1, (75, 150, 75))
    draw_vessel(img, (cx+210, cy+240), (cx+260, cy+310), 1, (75, 150, 75))
    draw_vessel(img, (cx+340, cy-10),  (cx+390, cy+40),  1, (72, 145, 72))
    draw_vessel(img, (cx-330, cy-10),  (cx-390, cy+30),  1, (72, 145, 72))

    # Optic disc (bright circular region)
    cv2.circle(img, (cx, cy), 45, (160, 220, 160), -1)
    cv2.circle(img, (cx, cy), 45, (120, 190, 120), 3)

    # Add Gaussian noise
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def create_ground_truth(h=565, w=565):
    """Create matching ground truth mask for demo evaluation."""
    gt = np.zeros((h, w), dtype=np.uint8)
    cx, cy = 280, 280

    def draw_v(p1, p2, t):
        cv2.line(gt, p1, p2, 255, t, cv2.LINE_AA)

    draw_v((cx, cy), (cx+180, cy-120), 5)
    draw_v((cx, cy), (cx-160, cy-100), 5)
    draw_v((cx, cy), (cx+150, cy+140), 4)
    draw_v((cx, cy), (cx-140, cy+130), 4)
    draw_v((cx+180, cy-120), (cx+280, cy-60),  3)
    draw_v((cx+180, cy-120), (cx+240, cy-200), 3)
    draw_v((cx-160, cy-100), (cx-260, cy-50),  3)
    draw_v((cx-160, cy-100), (cx-200, cy-200), 3)
    draw_v((cx+150, cy+140), (cx+270, cy+80),  2)
    draw_v((cx+150, cy+140), (cx+210, cy+240), 2)
    draw_v((cx+280, cy-60),  (cx+340, cy-10),  1)
    draw_v((cx+240, cy-200), (cx+310, cy-250), 1)
    draw_v((cx-260, cy-50),  (cx-330, cy-10),  1)
    draw_v((cx-200, cy-200), (cx-270, cy-250), 1)
    draw_v((cx+270, cy+80),  (cx+340, cy+120), 1)
    draw_v((cx+210, cy+240), (cx+260, cy+310), 1)
    draw_v((cx+340, cy-10),  (cx+390, cy+40),  1)
    draw_v((cx-330, cy-10),  (cx-390, cy+30),  1)
    cv2.circle(gt, (cx, cy), 45, 255, -1)
    return gt


def save_metrics_chart(paper_m, our_m, output_path):
    """Bar chart comparing paper vs our metrics — ready for IEEE report."""
    keys = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']
    paper_vals = [paper_m[k] for k in keys]
    our_vals   = [our_m[k]   for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d0d0d')
    ax.set_facecolor('#111111')

    bars1 = ax.bar(x - width/2, paper_vals, width,
                   label='Paper Method (Otsu only)',
                   color='#e05252', alpha=0.85, edgecolor='#ff7070')
    bars2 = ax.bar(x + width/2, our_vals, width,
                   label='Our Method (Dual Threshold)',
                   color='#52c0e0', alpha=0.85, edgecolor='#70d8ff')

    ax.set_ylabel('Score (%)', color='white', fontsize=12)
    ax.set_title('Performance Comparison: Paper vs Our Method',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.set_ylim(0, 105)
    ax.legend(facecolor='#222222', labelcolor='white', fontsize=10)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('white')
    ax.grid(axis='y', alpha=0.2, color='#555')

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                color='#ffaaaa', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                color='#aae8ff', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  ✓ Saved metrics chart: {output_path}")


def main():
    output_dir = Path('/home/claude/retinal_project/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*55)
    print("  RETINAL VESSEL SEGMENTATION — DEMO RUN")
    print("="*55)
    print("  (Using synthetic retinal image — replace with DRIVE dataset)")
    print()

    # 1. Create synthetic image + ground truth
    print("  [1/5] Creating synthetic retinal image...")
    demo_img = create_realistic_demo()
    gt_mask  = create_ground_truth()

    cv2.imwrite(str(output_dir / 'demo_input.png'),
                cv2.cvtColor(demo_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / 'demo_gt.png'), gt_mask)

    # 2. Preprocess
    print("  [2/5] Running preprocessing pipeline...")
    preprocessed, pre_stages = preprocess_pipeline(demo_img)

    # 3. Segment — Paper method
    print("  [3/5] Running segmentation (paper + our improvements)...")
    paper_mask, paper_stages = segment_pipeline_paper_only(preprocessed)
    our_mask,   our_stages   = segment_pipeline(preprocessed)

    # 4. Metrics
    print("  [4/5] Computing metrics...")
    paper_metrics = compute_metrics(paper_mask, gt_mask)
    our_metrics   = compute_metrics(our_mask,   gt_mask)

    print_metrics_table(paper_metrics, "Paper Method (Otsu only)")
    print_metrics_table(our_metrics,   "Our Method  (Dual Threshold + Top-Hat)")

    # 5. Save all figures
    print("  [5/5] Saving output figures...")

    save_result_figure(
        pre_stages, our_stages,
        output_dir / 'pipeline_visualization.png',
        our_metrics
    )

    save_comparison_figure(
        demo_img, paper_mask, our_mask, gt_mask,
        output_dir / 'method_comparison.png'
    )

    save_metrics_chart(
        paper_metrics, our_metrics,
        output_dir / 'metrics_comparison_chart.png'
    )

    # Save metrics as JSON (useful for report)
    results = {
        'paper_method':  paper_metrics,
        'our_method':    our_metrics,
        'improvement': {
            k: round(our_metrics[k] - paper_metrics[k], 2)
            for k in ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
            if k in our_metrics and k in paper_metrics
        }
    }
    with open(output_dir / 'results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  ALL DONE!")
    print("="*55)
    print(f"  Output saved to: {output_dir}/")
    print("  Files generated:")
    print("    📊 pipeline_visualization.png  — all processing stages")
    print("    📊 method_comparison.png        — paper vs our method")
    print("    📊 metrics_comparison_chart.png — bar chart for report")
    print("    📄 results.json                 — all metrics as data")
    print()
    print("  Next step: Replace synthetic image with DRIVE dataset images")
    print("  Download DRIVE from: https://drive.grand-challenge.org")
    print("="*55 + "\n")


if __name__ == '__main__':
    main()
