"""Experiment 2's ghosting knobs against the 16-sample reference, on whole captured runs (model.py's port of the
shaders, no network): python ghost.py <data folder> [runs...]

Per variant and ratio, over every frame from the 10th on:
  PSNR     of the final image (after the line pass)
  ghost    the mean |error| (0..255, luma) where the reference itself changed from the frame before along the motion
           by over 0.05 - animation, disocclusion, a surface replaced: where history lingers as a ghost
  temporal the frame-to-frame change along the motion the reference does not have (train.py's term), x 1000
  shimmer  at rest (no motion, the reference all but unchanged): the mean |luma change| from frame to frame
"""
import sys

import numpy as np
import torch

import model as M
import train as T

VARIANTS = [("x2", {}), ("greact", {"greact": 1}), ("drej", {"drej": 1}), ("dilate", {"dilate": 1}), ("vrej", {"vrej": 1}),
            ("minmax", {"minmax": 1}), ("all", {"greact": 1, "drej": 1, "dilate": 1, "vrej": 1, "minmax": 1})]
LW = torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)


def run(up, seq, ratio, chunk=60):
    """The whole sequence in chunks (the history and the previous frame carried across), per-frame stats."""
    hist, prev_f, prev_r = None, None, None
    up.prev_mda = None
    st = {"psnr": [], "ghost": [], "temporal": [], "shimmer": []}
    k = 0
    for c0 in range(0, len(seq), chunk):
        fr, meta = T.batch([seq[c0:c0 + chunk]], ratio, 1, len(seq[c0:c0 + chunk]), whole=True)
        with torch.no_grad():
            for i, f in enumerate(fr):
                out, final, _ = up(f["color"], f["depth"], f["mv"], f["jit"], hist, None, meta["out_size"], meta["tan_fov"],
                                   meta["diag"], meta["origin"], meta["full_size"])
                hist = out
                ref = f["ref"]
                if prev_f is not None and k >= 10:
                    uv, mo = T.out_uv_motion(f["mv"], meta["out_size"])
                    pu = uv - mo
                    ui = f["ui"] < 0.5
                    dref = ((ref - M.sample(prev_r, pu)) * LW).sum(1, keepdim=True).abs()
                    dout = ((final - M.sample(prev_f, pu)) * LW).sum(1, keepdim=True).abs()
                    err = ((final - ref) * LW).sum(1, keepdim=True).abs()
                    st["psnr"].append(T.psnr(final, ref))
                    ch = (dref > 0.05) & ui
                    if ch.sum() > 100:
                        st["ghost"].append(float(err[ch].mean()) * 255)
                    st["temporal"].append(float(((final - M.sample(prev_f, pu)) - (ref - M.sample(prev_r, pu))).abs().mean()) * 1000)
                    if float(f["mv"].abs().mean()) < 1e-5 and float(dref.mean()) < 0.002:
                        st["shimmer"].append(float(((final - prev_f) * LW).sum(1).abs().mean()) * 255)
                prev_f, prev_r = final, ref
                k += 1
    return {k_: (float(np.mean(v)) if v else float("nan")) for k_, v in st.items()}


def main():
    data = sys.argv[1]
    runs = sys.argv[2:] or ["walk", "menu", "val"]
    seqs = T.load_runs(data, runs)
    print(f"{len(seqs)} sequences, {sum(len(s) for s in seqs)} frames")
    for ratio in (2, 3):
        for name, opts in VARIANTS:
            up = M.Upscaler(False, opts).cuda()
            rs = [run(up, s, ratio) for s in seqs]
            m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
            print(f"{ratio}x {name:7s} PSNR {m['psnr']:.2f}  ghost {m['ghost']:.2f}  temporal {m['temporal']:.2f}  "
                  f"shimmer {m['shimmer']:.3f}", flush=True)


if __name__ == "__main__":
    main()
