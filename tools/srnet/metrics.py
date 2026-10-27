"""The measurements of measure.sh's runs: python metrics.py <out> [names...]

Per upscaler and mode (Performance 2x, Ultra Performance 3x):
  gpu      its own GPU time: the profile's rows from "ours: post" to "ours: upscaler done" (median of the frames
           after the first)
  still    against Native AA (nat/, DittoSR at Native AA): SSIM, PSNR (luma) and detail (mean |Laplacian| as a share
           of the reference's) in crops of scenery that does not move
  shimmer  frame-to-frame change at rest over 6 consecutive frames: the mean |luma difference| in the fence crop and
           over all the crops
  noise    fine noise in flat areas in motion: the mean |Laplacian| where the 8-pixel neighbourhood barely changes
           (the whole moving frame, the interface's corner left out)
  settle   after the game's menu closes (frame 1560): the mean |luma difference| to the settled frame (1600) at 1566,
           1570 and 1575, and the first frame from 1566 on below 1.0
Luma: BT.601, 0..255 (as ffmpeg's gray); the crops as the earlier sessions' scripts.
"""
import os
import re
import statistics
import sys

import numpy as np
from PIL import Image

STILL = [(1150, 200, 700, 450), (2650, 0, 1000, 650), (0, 450, 800, 300), (2100, 1000, 900, 700), (400, 1500, 900, 600)]
SHIMMER = {"face": (1820, 900, 240, 240), "fence": (1300, 200, 500, 400), "house": (2650, 0, 1000, 650),
           "sign": (1150, 200, 700, 450), "bush": (150, 1500, 700, 500)}


def luma(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return np.rint(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])


def crop(a, c):
    x, y, w, h = c
    return a[y:y + h, x:x + w]


def lap(a):
    return np.abs(4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])


def ssim(a, b):
    """SSIM with an 8x8 box window (ffmpeg's ssim filter's approach), luma 0..255."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def box(x):
        s = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
        return (s[8:, 8:] - s[:-8, 8:] - s[8:, :-8] + s[:-8, :-8]) / 64

    ma, mb = box(a), box(b)
    va, vb, cv = box(a * a) - ma * ma, box(b * b) - mb * mb, box(a * b) - ma * mb
    return float(np.mean((2 * ma * mb + c1) * (2 * cv + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2))))


def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))


def gpu_ms(log):
    frames, cur, on = [], 0.0, False
    for line in open(log, encoding="utf-8", errors="replace"):
        if "[gpuprof]" not in line:
            continue
        if "marks," in line:
            if on or cur:
                frames.append(cur)
            cur, on = 0.0, False
            continue
        m = re.match(r".*\[gpuprof\]\s+([\d.]+) ms\s+[\d.]+%\s+(.*)", line)
        if not m:
            continue
        what = m.group(2).strip()
        if what.startswith("ours: post"):
            on = True
            continue
        if what.startswith("ours: upscaler done"):
            if on:
                frames.append(cur)
            cur, on = 0.0, False
            continue
        if on:
            cur += float(m.group(1))
    frames = [f for f in frames if f > 0][1:]
    return statistics.median(frames) if frames else float("nan")


def main():
    out = sys.argv[1]
    names = sys.argv[2:] or ["gsr2", "x2", "net", "dexp", "dit"]
    ref = luma(os.path.join(out, "nat", "window.png"))
    print(f"{'mode':5s} {'upscaler':8s} {'gpu ms':>7s} | {'SSIM':>6s} {'PSNR':>6s} {'detail':>7s} | {'fence':>6s} {'all':>6s} | "
          f"{'noise':>6s} | settle 1566 1570 1575 (first < 1)")
    for pre, tag in (("performance", "2x"), ("ultra", "3x")):
        for n in names:
            t = os.path.join(out, f"t{pre}_{n}")
            row = f"{tag:5s} {n:8s} "
            try:
                row += f"{gpu_ms(os.path.join(out, f't{pre}_{n}.log')):7.3f} | "
                img = luma(os.path.join(t, "window.png"))
                ss = np.mean([ssim(crop(img, c), crop(ref, c)) for c in STILL])
                ps = np.mean([psnr(crop(img, c), crop(ref, c)) for c in STILL])
                de = np.mean([lap(crop(img, c)).mean() / lap(crop(ref, c)).mean() * 100 for c in STILL])
                row += f"{ss:6.4f} {ps:6.2f} {de:6.1f}% | "
                fs = [luma(os.path.join(t, f"fin_{i}.png")) for i in range(1596, 1602)]
                sh = {k: np.mean([np.abs(crop(a, c) - crop(b, c)).mean() for a, b in zip(fs, fs[1:])]) for k, c in SHIMMER.items()}
                row += f"{sh['fence']:6.3f} {np.mean(list(sh.values())):6.3f} | "
                mv = luma(os.path.join(out, f"m{pre}_{n}", "window.png"))[:1960]
                p = mv
                cen = p[8:-8, 8:-8]
                wide = np.abs(p[8:-8, 16:] - p[8:-8, :-16]) + np.abs(p[16:, 8:-8] - p[:-16, 8:-8])
                lp = np.abs(4 * cen - p[8:-8, 7:-9] - p[8:-8, 9:-7] - p[7:-9, 8:-8] - p[9:-7, 8:-8])
                row += f"{lp[wide <= 6].mean():6.2f} | "
                rr = os.path.join(out, f"r{pre}_{n}")
                settled = luma(os.path.join(rr, "fin_1600.png"))[60:1960]
                d = {i: np.abs(luma(os.path.join(rr, f"fin_{i}.png"))[60:1960] - settled).mean() for i in range(1566, 1581)}
                first = next((i for i in range(1566, 1581) if d[i] < 1.0), None)
                row += f"       {d[1566]:4.2f} {d[1570]:4.2f} {d[1575]:4.2f} ({first if first else '>1580'})"
            except (OSError, KeyError, ValueError) as e:
                row += f"(missing: {e})"
            print(row, flush=True)


if __name__ == "__main__":
    main()
