"""
N2V Evaluation & Metrics
========================
Computes PSNR and SSIM between noisy and denoised images (and optionally
against clean ground-truth if available).  Also produces side-by-side
visual comparison tiles.

Usage:
    python n2v_eval.py --noisy_dir ./noisy --denoised_dir ./denoised
    python n2v_eval.py --noisy_dir ./noisy --denoised_dir ./denoised --clean_dir ./clean
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(ref: np.ndarray, img: np.ndarray) -> dict:
    """Return PSNR and SSIM between two float32 arrays normalised to [0,1]."""
    ref_f = ref.astype(np.float32) / 255.0
    img_f = img.astype(np.float32) / 255.0
    p = psnr(ref_f, img_f, data_range=1.0)
    s = ssim(ref_f, img_f, data_range=1.0)
    return {"psnr": p, "ssim": s}


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


# ─────────────────────────────────────────────
# Visual comparison tile
# ─────────────────────────────────────────────
def save_comparison(noisy: np.ndarray,
                    denoised: np.ndarray,
                    out_path: str,
                    clean: np.ndarray = None,
                    title: str = ""):
    cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    axes[0].imshow(noisy,    cmap="gray", vmin=0, vmax=255); axes[0].set_title("Noisy");    axes[0].axis("off")
    axes[1].imshow(denoised, cmap="gray", vmin=0, vmax=255); axes[1].set_title("Denoised"); axes[1].axis("off")
    if clean is not None:
        axes[2].imshow(clean, cmap="gray", vmin=0, vmax=255); axes[2].set_title("Clean GT"); axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
def evaluate(args):
    noisy_dir    = Path(args.noisy_dir)
    denoised_dir = Path(args.denoised_dir)
    clean_dir    = Path(args.clean_dir) if args.clean_dir else None
    out_dir      = Path(args.output_dir)
    tile_dir     = out_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    noisy_files = sorted([p for p in noisy_dir.rglob("*") if p.suffix.lower() in exts])
    print(f"Evaluating {len(noisy_files)} images …")

    records = []
    vis_count = 0

    for nf in tqdm(noisy_files):
        df = denoised_dir / nf.name
        if not df.exists():
            print(f"  [SKIP] No denoised match for {nf.name}")
            continue

        noisy   = load_gray(nf)
        denoised = load_gray(df)

        row = {"filename": nf.name}

        if clean_dir:
            cf = clean_dir / nf.name
            if cf.exists():
                clean = load_gray(cf)
                row.update({f"noisy_{k}":    v for k, v in compute_metrics(clean, noisy).items()})
                row.update({f"denoised_{k}": v for k, v in compute_metrics(clean, denoised).items()})
            else:
                clean = None
        else:
            clean = None

        records.append(row)

        # Save the first N visual comparisons
        if vis_count < args.num_visuals:
            title_str = f"{nf.name}"
            if clean is not None:
                n_psnr = row.get("noisy_psnr", 0)
                d_psnr = row.get("denoised_psnr", 0)
                title_str += f"  |  Noisy PSNR={n_psnr:.2f} dB → Denoised PSNR={d_psnr:.2f} dB"
            save_comparison(noisy, denoised,
                            str(tile_dir / f"compare_{vis_count:04d}.png"),
                            clean=clean, title=title_str)
            vis_count += 1

    # Summary statistics
    df_all = pd.DataFrame(records)
    print("\n── Metric Summary ──────────────────────────────────────────")
    print(df_all.describe().to_string())

    csv_path = out_dir / "metrics.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nMetrics saved → {csv_path}")

    # Distribution plots
    if clean_dir:
        _plot_metric_distributions(df_all, out_dir)

    print(f"Visual comparisons saved → {tile_dir}/")


def _plot_metric_distributions(df, out_dir):
    for metric in ["psnr", "ssim"]:
        n_col = f"noisy_{metric}"
        d_col = f"denoised_{metric}"
        if n_col not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(df[n_col],    bins=40, alpha=0.6, label=f"Noisy {metric.upper()}", color="salmon")
        plt.hist(df[d_col],    bins=40, alpha=0.6, label=f"Denoised {metric.upper()}", color="steelblue")
        mn, md = df[n_col].mean(), df[d_col].mean()
        plt.axvline(mn, color="red",  linestyle="--", linewidth=1, label=f"Noisy mean={mn:.2f}")
        plt.axvline(md, color="blue", linestyle="--", linewidth=1, label=f"Denoised mean={md:.2f}")
        plt.xlabel(metric.upper()); plt.ylabel("Count")
        plt.title(f"Distribution of {metric.upper()}")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(out_dir / f"dist_{metric}.png"), dpi=150)
        plt.close()
        print(f"  {metric.upper()}: {mn:.3f} → {md:.3f}  (Δ={md-mn:+.3f})")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_dir",    required=True)
    p.add_argument("--denoised_dir", required=True)
    p.add_argument("--clean_dir",    default=None,       help="Optional GT clean images")
    p.add_argument("--output_dir",   default="./eval_results")
    p.add_argument("--num_visuals",  type=int, default=20, help="Number of comparison tiles to save")
    evaluate(p.parse_args())
