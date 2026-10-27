"""Trains the tiny network (model.py) on the runtime's captures (capture.sh, srcap.py).

python train.py <data folder> [--steps N] [--out srnet.pt] [--val val]

Each step: a batch of crops (384 x 384 output pixels) of sequences of WARM + TRAIN consecutive frames at 2x or 3x.
The first WARM frames build the history without gradients, then TRAIN frames are unrolled with them (the history
and the network's state carry the gradient from frame to frame). The loss, on the final image (after the line pass)
against the 16-sample reference, the interface left out: Charbonnier + its gradients' L1 (detail) + the temporal
term - the frame-to-frame change (along the motion) that the reference does not have (shimmer, flicker).
"""
import argparse
import os
import queue
import random
import struct
import threading
import time

import numpy as np
import torch

import model as M
import srcap

CROP, WARM, TRAIN, BATCH = 384, 4, 6, 8
W_DETAIL, W_TEMPORAL = 0.5, 1.0
BORDER = 24  # (output pixels at the crop's edge: its history clamps there)


class Frames:
    """A capture file: its header and the planes' offsets; crops read from the file as needed (memory maps of the
    whole set, 60 GB, filled the RAM)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = f.read(64)
        self.path = path
        self.size = os.path.getsize(path)
        _, self.ow, self.oh = struct.unpack_from("<3I", hdr, 4)
        self.proj = struct.unpack_from("<4f", hdr, 16)
        off = 64
        self.ref_off = off
        off += self.ow * self.oh * 4
        self.lr = {}
        for r in range(2):
            lw, lh = struct.unpack_from("<2I", hdr, 32 + 16 * r)
            jit = struct.unpack_from("<2f", hdr, 40 + 16 * r)
            n = lw * lh
            self.lr[2 + r] = (lw, lh, jit, off, off + n * 4, off + n * 8)
            off += n * 12

    def complete(self):
        lw, lh, _, _, _, om = self.lr[3]
        return self.size >= om + lw * lh * 4

    def _rows(self, f, off, width, y, h, x, w, dtype, ch):
        """Rows y..y+h of a plane width pixels wide (ch values of dtype each), columns x..x+w."""
        bpp = np.dtype(dtype).itemsize * ch
        f.seek(off + y * width * bpp)
        a = np.frombuffer(f.read(h * width * bpp), dtype).reshape(h, width, ch)
        return a[:, x:x + w]

    def ref(self, x, y, w, h):
        with open(self.path, "rb") as f:
            return self._rows(f, self.ref_off, self.ow, y, h, x, w, np.uint8, 4)

    def low(self, r, x, y, w, h):
        lw, lh, jit, oc, od, om = self.lr[r]
        with open(self.path, "rb") as f:
            c = self._rows(f, oc, lw, y, h, x, w, np.uint8, 4)
            d = self._rows(f, od, lw, y, h, x, w, np.float32, 1)[..., 0]
            m = self._rows(f, om, lw, y, h, x, w, np.float16, 2)
        return c, d, m, jit


def load_runs(root, names):
    seqs = []
    for n in names:
        for s in srcap.list_frames(os.path.join(root, n)):
            fs = [Frames(p) for p in s]
            fs = [f for f in fs if f.complete()]  # (the last one of a run can be cut short: the game exited)
            if fs:
                seqs.append(fs)
    return seqs


def batch(seqs, ratio, n, length, crop=CROP, dev="cuda", whole=False):
    """n crops of `length` consecutive frames: tensors per frame (colour, depth, motion, jitter, reference, mask)."""
    out = []
    for _ in range(n):
        s = random.choice([s for s in seqs if len(s) >= length])
        t0 = random.randrange(len(s) - length + 1)
        f0 = s[t0]
        if whole:
            ox = oy = 0
            cw, ch = f0.ow, f0.oh
        else:
            cw = ch = crop
            ox = random.randrange((f0.ow - cw) // 6 + 1) * 6
            oy = random.randrange((f0.oh - ch) // 6 + 1) * 6
        out.append((s[t0:t0 + length], ox, oy, cw, ch))
    frames = []
    for k in range(length):
        cs, ds, ms, js, rs, us = [], [], [], [], [], []
        for seq, ox, oy, cw, ch in out:
            f = seq[k]
            lw, lh = f.lr[ratio][0], f.lr[ratio][1]
            c, d, m, jit = f.low(ratio, ox // ratio, oy // ratio, cw // ratio, ch // ratio)
            ref = f.ref(ox, oy, cw, ch)
            cs.append(np.ascontiguousarray(c[..., :3]))
            ds.append(np.ascontiguousarray(d))
            # (the motion in the crop's uv units)
            ms.append(np.nan_to_num(np.ascontiguousarray(m).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) * np.array([lw / (cw // ratio), lh / (ch // ratio)], np.float32))
            js.append(jit)
            rs.append(np.ascontiguousarray(ref[..., :3]))
            us.append(np.ascontiguousarray(np.maximum(ref[..., 3], 0)))
        t = lambda a, dt: torch.from_numpy(np.stack(a)).to(dev, non_blocking=True).to(dt)
        frames.append({
            "color": t(cs, torch.float32).permute(0, 3, 1, 2) / 255.0,
            "depth": t(ds, torch.float32).unsqueeze(1),
            "mv": t(ms, torch.float32).permute(0, 3, 1, 2),
            "jit": torch.tensor(js, device=dev, dtype=torch.float32),
            "ref": t(rs, torch.float32).permute(0, 3, 1, 2) / 255.0,
            "ui": t(us, torch.float32).unsqueeze(1) / 255.0,
        })
    f0 = out[0][0][0]
    lw, lh = f0.lr[ratio][0], f0.lr[ratio][1]
    meta = {
        "out_size": (out[0][3], out[0][4]),
        "origin": torch.tensor([[o[1], o[2]] for o in out], device=dev, dtype=torch.float32),
        "full_size": (f0.ow, f0.oh),
        "tan_fov": torch.tensor([(lw / lh) / s[0][0].proj[1] for s in out], device=dev, dtype=torch.float32),
        # (the depth clip's threshold grows with the render size's diagonal: the 4K one, where it is played)
        "diag": 2.0 * float(np.hypot(lw, lh)),
    }
    return frames, meta


def out_uv_motion(mv, size):
    """The motion vectors at the output pixels (bilinear from the render size), B x H x W x 2."""
    W, H = size
    B = mv.shape[0]
    ys = (torch.arange(H, device=mv.device) + 0.5) / H
    xs = (torch.arange(W, device=mv.device) + 0.5) / W
    uv = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).expand(B, H, W, 2)
    return uv, M.sample(mv, uv).permute(0, 2, 3, 1)


def grad(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def run_seq(net, frames, meta, train_from):
    """Unrolls the upscaler over the frames; returns the per-frame finals from train_from on (and the loss parts)."""
    hist = state = None
    prev_final = prev_ref = None
    parts = torch.zeros(3, device=frames[0]["color"].device)
    finals = []
    for k, fr in enumerate(frames):
        grad_on = k >= train_from and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_on):
            out, final, state_n = net(fr["color"], fr["depth"], fr["mv"], fr["jit"], hist, state, meta["out_size"],
                                      meta["tan_fov"], meta["diag"], meta["origin"], meta["full_size"])
            if k >= train_from:
                ref = fr["ref"]
                wmask = (1.0 - fr["ui"]).clamp(0, 1)
                wmask[..., :BORDER, :] = 0
                wmask[..., -BORDER:, :] = 0
                wmask[..., :BORDER] = 0
                wmask[..., -BORDER:] = 0
                den = wmask.sum() * 3 + 1
                e = final - ref
                parts[0] = parts[0] + (torch.sqrt(e * e + 1e-6) * wmask).sum() / den
                gy, gx = grad(final)
                ry, rx = grad(ref)
                parts[1] = parts[1] + ((gy - ry).abs() * wmask[..., 1:, :]).sum() / den + ((gx - rx).abs() * wmask[..., :, 1:]).sum() / den
                if prev_final is not None:
                    uv, mo = out_uv_motion(fr["mv"], meta["out_size"])
                    pu = uv - mo
                    vis = ((pu >= 0) & (pu <= 1)).all(-1).unsqueeze(1).float() * wmask
                    dt_out = final - M.sample(prev_final, pu)
                    dt_ref = ref - M.sample(prev_ref, pu)
                    parts[2] = parts[2] + ((dt_out - dt_ref).abs() * vis).sum() / (vis.sum() * 3 + 1)
                prev_final, prev_ref = final, ref
                finals.append(final)
        hist = out if grad_on else out.detach()
        state = state_n if state_n is None or grad_on else state_n.detach()
    n = len(frames) - train_from
    return finals, parts / n


def psnr(a, b):
    return float(10 * torch.log10(1.0 / ((a - b) ** 2).mean().clamp_min(1e-10)))


def evaluate(net, seqs, ratio, frames=24, dev="cuda"):
    """The whole first validation sequence (frames 30..30+frames, the history built over 30 before): PSNR of the final
    image, and the temporal error (frame-to-frame change along the motion beyond the reference's)."""
    net.eval()
    s = seqs[0]
    fr, meta = batch([s[:30 + frames]], ratio, 1, 30 + frames, dev=dev, whole=True)
    with torch.no_grad():
        finals, parts = run_seq(net, fr, meta, 30)
        p = sum(psnr(f, fr[30 + i]["ref"]) for i, f in enumerate(finals)) / len(finals)
    net.train()
    return p, float(parts[0]), float(parts[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--runs", default="walk,down,menu,diag,zigzag")
    ap.add_argument("--val", default="val")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="srnet.pt")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"
    seqs = load_runs(a.data, a.runs.split(","))
    val = load_runs(a.data, [a.val])
    print(f"{sum(len(s) for s in seqs)} training frames in {len(seqs)} sequences, {sum(len(s) for s in val)} for validation")
    net = M.Upscaler(use_net=True).to(dev)
    base = M.Upscaler(use_net=False).to(dev)
    print(f"parameters: {sum(p.numel() for p in net.parameters())}")
    for r in (2, 3):
        print(f"Experiment 2 alone at {r}x: PSNR %.2f, Charbonnier %.4f, temporal %.4f" % evaluate(base, val, r))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time()
    run = torch.zeros(3, device=dev)
    # (the batches read ahead by a thread: the reads overlap the training)
    q = queue.Queue(maxsize=3)

    def reader():
        for step in range(1, a.steps + 1):
            q.put(batch(seqs, 2 + (step & 1), BATCH, WARM + TRAIN, dev=dev))

    threading.Thread(target=reader, daemon=True).start()
    for step in range(1, a.steps + 1):
        frames, meta = q.get()
        _, parts = run_seq(net, frames, meta, WARM)
        loss = parts[0] + W_DETAIL * parts[1] + W_TEMPORAL * parts[2]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run += parts.detach()
        if step % 100 == 0:
            r = (run / 100).tolist()
            run.zero_()
            print(f"step {step}: charb {r[0]:.4f} detail {r[1]:.4f} temporal {r[2]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if step % 1000 == 0 or step == a.steps:
            for rr in (2, 3):
                print(f"  val {rr}x: PSNR %.2f, Charbonnier %.4f, temporal %.4f" % evaluate(net, val, rr), flush=True)
            torch.save(net.net.state_dict(), a.out)


if __name__ == "__main__":
    main()
