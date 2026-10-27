"""The trained network against Experiment 2 alone on a validation sequence, in FP32 and with the network in FP16
(weights and activations rounded to half precision, as the shader runs it), and a side-by-side crop.

python evaluate.py <data folder> <weights.pt> [--run val] [--png out.png]
"""
import argparse

import numpy as np
import torch
from PIL import Image

import model as M
import train as T


class Half(torch.nn.Module):
    """The network in half precision: inputs, weights, activations and outputs rounded to FP16."""
    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, x):
        return self.net(x.half()).float()


def run(up, seq, ratio, start, warm, n):
    fr, meta = T.batch([seq[start:start + warm + n]], ratio, 1, warm + n, whole=True)
    with torch.no_grad():
        finals, parts = T.run_seq(up, fr, meta, warm)
    p = np.mean([T.psnr(f, fr[warm + i]["ref"]) for i, f in enumerate(finals)])
    return p, float(parts[2]), finals[-1], fr[-1]["ref"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("weights")
    ap.add_argument("--run", default="val")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    seq = T.load_runs(a.data, [a.run])[0]
    base = M.Upscaler(False).cuda()
    net = M.Upscaler(True).cuda()
    net.net.load_state_dict(torch.load(a.weights))
    half = M.Upscaler(True).cuda()
    half.net.load_state_dict(torch.load(a.weights))
    half.net = Half(half.net)
    crops = []
    for ratio in (2, 3):
        for start in (60, 300, 500):
            rb = run(base, seq, ratio, start, 30, 30)
            rn = run(net, seq, ratio, start, 30, 30)
            rh = run(half, seq, ratio, start, 30, 30)
            print(f"{ratio}x from {start}: Experiment 2 {rb[0]:.2f} dB temporal {rb[1]:.4f} | net {rn[0]:.2f} dB "
                  f"temporal {rn[1]:.4f} | net FP16 {rh[0]:.2f} dB temporal {rh[1]:.4f}")
            if start == 300:
                crops.append((ratio, rb[2], rn[2], rn[3]))
    if a.png:
        rows = []
        for ratio, b, n, r in crops:
            # (the images are stored bottom-up: flipped for viewing; a crop around the player)
            def c(x):
                im = (x[0].permute(1, 2, 0).flip(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                return im[380:680, 760:1160]
            rows.append(np.concatenate([c(b), c(n), c(r)], 1))
        Image.fromarray(np.concatenate(rows, 0)).resize((2400, 1200), Image.NEAREST).save(a.png)


if __name__ == "__main__":
    main()
