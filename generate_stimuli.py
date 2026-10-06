#!/usr/bin/env python3
"""
Usage:
    pip install pillow numpy pandas
    python generate_stimuli.py --outdir reproduced_stimuli --per-type 50 --seed 42
    python generate_stimuli.py --outdir reproduced_stimuli --per-type 20 --seed 123 --append
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# --------- Fixed geometry based on spec ---------
N_POINTS = 10
BAR_WIDTH = 31
SPACING = 31
FIRST_CENTER = 15
X_AXIS_LEN = 309
DOT_DIAM = 10
HEIGHTS = list(range(70, 171, 10))  # 70..170 by 10

# Rendering padding (canvas margins around the 0..309 range)
PAD_L = 10
PAD_R = 10
PAD_T = 10
PAD_B = 10

# Canvas size
Y_MAX = 200
IMG_W = X_AXIS_LEN + PAD_L + PAD_R
IMG_H = Y_MAX + PAD_T + PAD_B

# X positions (centers) for the 10 items
X_CENTERS = [FIRST_CENTER + i * SPACING for i in range(N_POINTS)]
SERIAL = np.arange(1, N_POINTS + 1)


def pearson_r2(y_vals):
    y = np.array(y_vals, dtype=float)
    r = np.corrcoef(SERIAL, y)[0, 1]
    return float(r), float(r**2)


def regression_slope(y_vals):
    """OLS slope of y on x (X_CENTERS). Units: change in y (px) per pixel in x."""
    y = np.array(y_vals, dtype=float)
    x = np.array(X_CENTERS, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def generate_candidate_positive(rng, r2_min, r2_max):
    """Return a qualifying positive sequence, or None."""
    y = [rng.choice(HEIGHTS) for _ in range(N_POINTS)]
    r, r2 = pearson_r2(y)
    if r > 0 and r2_min <= r2 <= r2_max:
        return y, r, r2
    return None


def draw_axis(draw: ImageDraw.ImageDraw, origin_px, x_end_px):
    draw.line([origin_px, (x_end_px, origin_px[1])], width=8, fill=(0, 0, 0))


def make_canvas():
    """Return (img, draw, origin), where origin is the (x,y) pixel of x-axis start at baseline."""
    img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    origin = (PAD_L, IMG_H - PAD_B)  # baseline y
    draw_axis(d, origin, PAD_L + X_AXIS_LEN)
    return img, d, origin


def to_px(x, y, origin):
    """Map data coords (x in px from 0..309, y in px up from baseline) to image pixel coords.
    Image origin (0,0) is top-left; y increases downward, so we subtract y from baseline.
    """
    return (PAD_L + x, origin[1] - y)


def draw_bars(y_vals, fname):
    img, d, origin = make_canvas()
    for xc, h in zip(X_CENTERS, y_vals):
        left_x = xc - BAR_WIDTH // 2
        right_x = left_x + BAR_WIDTH
        x0, y0 = to_px(left_x, h, origin)
        x1, y1 = to_px(right_x, 0, origin)  # bottom at baseline
        d.rectangle([x0, y0, x1, y1], fill=(180, 180, 180), outline=(0, 0, 0), width=1)
    img.save(fname, "PNG")


def draw_scatter(y_vals, fname):
    # Create a larger image for supersampling (antialiasing)
    scale = 4
    big_img = Image.new("RGB", (IMG_W * scale, IMG_H * scale), (255, 255, 255))
    big_d = ImageDraw.Draw(big_img)
    
    # Scale up the origin and axis
    big_origin = (PAD_L * scale, (IMG_H - PAD_B) * scale)
    big_d.line([big_origin, ((PAD_L + X_AXIS_LEN) * scale, big_origin[1])], width=8 * scale, fill=(0, 0, 0))
    
    # Draw scaled dots
    r = (DOT_DIAM // 2) * scale
    for xc, h in zip(X_CENTERS, y_vals):
        cx = (PAD_L + xc) * scale
        cy = big_origin[1] - (h * scale)
        big_d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 50, 50), outline=(0, 0, 0), width=1 * scale)
    
    # Downsample for antialiasing
    img = big_img.resize((IMG_W, IMG_H), Image.LANCZOS)
    img.save(fname, "PNG")


def ensure_dirs(root):
    for sub in ["bars_pos", "bars_neg", "scatter_pos", "scatter_neg"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def load_existing_manifest(path_csv):
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    return pd.DataFrame(columns=[
        "chart", "slope", "index", "filename", "r", "r2", "y_values",
        "regression_slope",
    ])


def existing_max_index(df, chart, slope):
    if df.empty:
        return 0
    sub = df[(df["chart"] == chart) & (df["slope"] == slope)]
    if sub.empty:
        return 0
    return int(sub["index"].max())


def existing_positive_signatures(df, chart):
    """Track only positives to avoid duplicates."""
    sigs = set()
    if df.empty:
        return sigs
    sub = df[(df["chart"] == chart) & (df["slope"] == "positive")]
    for _, row in sub.iterrows():
        try:
            y = tuple(json.loads(row["y_values"]))
        except Exception:
            continue
        sigs.add(y)
    return sigs


def collect(chart_type, add_count, rng, root, df_existing, r2_min=0.75, r2_max=0.85):
    """Generate 'add_count' NEW positive stimuli for chart_type and mirror for negative.
       Returns new records (list of dicts)."""
    records = []
    start_index = existing_max_index(df_existing, chart_type, "positive") + 1
    sigs = existing_positive_signatures(df_existing, chart_type)

    attempts = 0
    collected = 0
    max_attempts = 400000

    while collected < add_count and attempts < max_attempts:
        attempts += 1
        cand = generate_candidate_positive(rng, r2_min, r2_max)
        if not cand:
            continue
        y, r, r2 = cand
        sig = tuple(y)
        if sig in sigs:
            continue
        sigs.add(sig)

        # Positive path & draw
        if chart_type == "bars":
            fname_pos = os.path.join(root, "bars_pos", f"bars_pos_{start_index:03d}.png")
            draw_bars(y, fname_pos)
        else:
            fname_pos = os.path.join(root, "scatter_pos", f"scatter_pos_{start_index:03d}.png")
            draw_scatter(y, fname_pos)

        records.append({
            "chart": chart_type,
            "slope": "positive",
            "index": start_index,
            "filename": os.path.basename(fname_pos),
            "r": round(float(r), 9),
            "r2": round(float(r2), 9),
            "y_values": json.dumps(y),
            "regression_slope": round(regression_slope(y), 9),
        })

        # Decreasing graphs are exact mirrors of their increasing partners.
        y_neg = list(reversed(y))
        r_neg, r2_neg = pearson_r2(y_neg)
        if chart_type == "bars":
            fname_neg = os.path.join(root, "bars_neg", f"bars_neg_{start_index:03d}.png")
            draw_bars(y_neg, fname_neg)
        else:
            fname_neg = os.path.join(root, "scatter_neg", f"scatter_neg_{start_index:03d}.png")
            draw_scatter(y_neg, fname_neg)

        records.append({
            "chart": chart_type,
            "slope": "negative",
            "index": start_index,
            "filename": os.path.basename(fname_neg),
            "r": round(float(r_neg), 9),
            "r2": round(float(r2_neg), 9),
            "y_values": json.dumps(y_neg),
            "regression_slope": round(regression_slope(y_neg), 9),
        })

        start_index += 1
        collected += 1

    if collected < add_count:
        raise RuntimeError(
            f"Stopped after {attempts} attempts; only collected {collected} of {add_count} for {chart_type}. "
            "Consider widening the r^2 window or changing the RNG seed."
        )
    return records


def save_manifests(root, df_all):
    csv_path = os.path.join(root, "Stimuli_Info.csv")
    df_all.to_csv(csv_path, index=False)

    full_manifest = df_all.to_dict(orient="records")
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(full_manifest, f, indent=2)

    minimal = [
        {
            "chart": row["chart"],
            "slope": row["slope"],
            "filename": row["filename"],
            "r": float(row["r"]),
            "r2": float(row["r2"]),
        }
        for _, row in df_all.iterrows()
    ]
    with open(os.path.join(root, "stimuli_list.json"), "w") as f:
        json.dump(minimal, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Generate pixel-exact bar/scatter stimuli and manifests.")
    ap.add_argument("--outdir", default="reproduced_stimuli", help="Output directory.")
    ap.add_argument("--per-type", type=int, default=50,
                    help="How many POSITIVE sequences per chart type to add (negatives are mirrored).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing manifest (continue indices).")
    args = ap.parse_args()

    root = args.outdir
    ensure_dirs(root)

    manifest_csv = os.path.join(root, "Stimuli_Info.csv")
    if args.append:
        df_existing = load_existing_manifest(manifest_csv)
    else:
        df_existing = pd.DataFrame(columns=[
            "chart", "slope", "index", "filename", "r", "r2", "y_values",
            "regression_slope",
        ])

    new_records = []

    print("Generating graphs with r^2 in [0.75, 0.85]...")

    # These offsets reproduce the independent streams used for the pilot files.
    rng_bars = random.Random(args.seed)
    new_records += collect("bars", args.per_type, rng_bars, root, df_existing)

    rng_scatter = random.Random(args.seed + 999)
    new_records += collect("scatter", args.per_type, rng_scatter, root, df_existing)

    # Merge & save manifests
    df_all = pd.concat([df_existing, pd.DataFrame(new_records)], ignore_index=True)
    save_manifests(root, df_all)

    print(f"Done.\n - Output dir: {root}\n - Total rows in Stimuli_Info.csv: {len(df_all)}")


if __name__ == "__main__":
    main()
