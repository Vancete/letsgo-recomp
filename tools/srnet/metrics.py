"""The measurements of measure.sh's runs: python metrics.py <out> [names...]

Per upscaler and mode (those measured: Quality 1.5x ... Ultra Performance 3x), and native 4K (no upscaler) first:
  upsc ms  its own GPU time: the profile's rows from "ours: post" to "ours: upscaler done" (median of the frames
           after the first)
  frame    the whole frame's GPU time (first to last mark, median)
  still    against native 4K (nat/: no upscaler, SMAA): SSIM, PSNR (luma) and detail (mean |Laplacian| as a share
           of the reference's) in crops of scenery that does not move
  shimmer  frame-to-frame change at rest over 6 consecutive frames: the mean |luma difference| in the fence crop and
           over all the crops
  noise    fine noise in flat areas in motion: the mean |Laplacian| where the 8-pixel neighbourhood barely changes
           (the whole moving frame, the interface's corner left out)
  settle   (when measured) after the game's menu closes (frame 1560): the mean |luma difference| to the settled frame
           (1600) at 1566, 1570 and 1575
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
    """(the upscaler's time, the scene's main pass) per profiled frame, the first left out."""
    frames, cur, on, scene = [], 0.0, False, 0.0
    for line in open(log, encoding="utf-8", errors="replace"):
        if "[gpuprof]" not in line:
            continue
        if "marks," in line:
            cur, on, scene = 0.0, False, 0.0
            continue
        m = re.match(r".*\[gpuprof\]\s+([\d.]+) ms\s+[\d.]+%\s+(.*)", line)
        if not m:
            continue
        what = m.group(2).strip()
        dm = re.match(r"pass: 1 color \d+x\d+ .* (\d+) draws", what)
        if dm and int(dm.group(1)) > 100:
            scene = max(scene, float(m.group(1)))
        if what.startswith("ours: post"):
            on = True
            continue
        if what.startswith("ours: upscaler done"):
            if on and scene > 0:
                frames.append((cur, scene))
            cur, on = 0.0, False
            continue
        if on:
            cur += float(m.group(1))
    return [f for f in frames if f[0] > 0][1:]


def frame_ms(log):
    """The frame's GPU time per profiled frame (first to last mark), median, the first frame left out."""
    t = [float(m.group(1)) for m in re.finditer(r"marks, ([\d.]+) ms first to last", open(log, encoding="utf-8", errors="replace").read())]
    t = t[1:] if len(t) > 2 else t
    return statistics.median(t) if t else float("nan")


def stability(d):
    """(the fence's shimmer, all the crops') over the 6 consecutive images at rest in d."""
    fs = [luma(os.path.join(d, f"fin_{i}.png")) for i in range(1596, 1602)]
    sh = {k: np.mean([np.abs(crop(a, c) - crop(b, c)).mean() for a, b in zip(fs, fs[1:])]) for k, c in SHIMMER.items()}
    return sh["fence"], float(np.mean(list(sh.values())))


def noise(path):
    p = luma(path)[:1960]
    cen = p[8:-8, 8:-8]
    wide = np.abs(p[8:-8, 16:] - p[8:-8, :-16]) + np.abs(p[16:, 8:-8] - p[:-16, 8:-8])
    lp = np.abs(4 * cen - p[8:-8, 7:-9] - p[8:-8, 9:-7] - p[7:-9, 8:-8] - p[9:-7, 8:-8])
    return float(lp[wide <= 6].mean())


MODES = {"quality": "1.5x", "balanced": "1.7x", "performance": "2x", "ultra": "3x"}


def main():
    out = sys.argv[1]
    names = sys.argv[2:] or ["fsr3", "ditto", "fsr4", "net"]
    ref = luma(os.path.join(out, "nat", "window.png"))
    print(f"{'mode':5s} {'upscaler':8s} {'upsc ms':>7s} {'frame':>6s} | {'SSIM':>6s} {'PSNR':>6s} {'detail':>7s} | "
          f"{'fence':>6s} {'all':>6s} | {'noise':>6s}" + (" | settle 1566 1570 1575" if os.path.isdir(os.path.join(out, "rperformance_" + names[0])) else ""))
    row = f"{'1x':5s} {'native':8s} {'-':>7s} {frame_ms(os.path.join(out, 'pnat.log')):6.2f} | {1:6.4f} {99:6.2f} {100:6.1f}% | "
    fe, al = stability(os.path.join(out, "nat"))
    print(row + f"{fe:6.3f} {al:6.3f} | {noise(os.path.join(out, 'mnat', 'window.png')):6.2f}")
    for pre, tag in MODES.items():
        if not any(os.path.exists(os.path.join(out, f"p{pre}_{n}.log")) for n in names):
            continue
        # (the GPU's clock changes from run to run: each upscaler's time scaled by the scene's main pass - the same work
        # in every run of a mode - to the fastest run's)
        prof = {n: gpu_ms(os.path.join(out, f"p{pre}_{n}.log")) if os.path.exists(os.path.join(out, f"p{pre}_{n}.log")) else []
                for n in names}
        fastest = min((sc for fr in prof.values() for _, sc in fr), default=float("nan"))
        for n in names:
            t = os.path.join(out, f"t{pre}_{n}")
            row = f"{tag:5s} {n:8s} "
            try:
                g = [u * fastest / sc for u, sc in prof[n]]
                row += f"{statistics.median(g) if g else float('nan'):7.3f} {frame_ms(os.path.join(out, f'p{pre}_{n}.log')):6.2f} | "
                img = luma(os.path.join(t, "window.png"))
                ss = np.mean([ssim(crop(img, c), crop(ref, c)) for c in STILL])
                ps = np.mean([psnr(crop(img, c), crop(ref, c)) for c in STILL])
                de = np.mean([lap(crop(img, c)).mean() / lap(crop(ref, c)).mean() * 100 for c in STILL])
                row += f"{ss:6.4f} {ps:6.2f} {de:6.1f}% | "
                fe, al = stability(t)
                row += f"{fe:6.3f} {al:6.3f} | {noise(os.path.join(out, f'm{pre}_{n}', 'window.png')):6.2f}"
                rr = os.path.join(out, f"r{pre}_{n}")
                if os.path.isdir(rr):
                    settled = luma(os.path.join(rr, "fin_1600.png"))[60:1960]
                    d = {i: np.abs(luma(os.path.join(rr, f"fin_{i}.png"))[60:1960] - settled).mean() for i in (1566, 1570, 1575)}
                    row += f" |        {d[1566]:4.2f} {d[1570]:4.2f} {d[1575]:4.2f}"
            except (OSError, KeyError, ValueError) as e:
                row += f"(missing: {e})"
            print(row, flush=True)


if __name__ == "__main__":
    main()
