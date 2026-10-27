"""Reader of the teacher captures (LARECOMP_EXPCAP, runtime/nvn/vulkan/exec.c expcap_run; LOCAL research only: the
captures and anything trained on them stay on this machine).

One file per real frame, f<present>.bin: a 64-byte header ("EXP1", present, output width and height, render width and
height, the jitter in render pixels, the projection P00 P11 A B), then the planes: DittoSR Experiment's output RGBA8 at
the output size, and at the render size the colour RGBA8 (alpha: the interface mask), the depth R32F and the motion
vectors RG16F (uv now - uv before). Frames reads them as train.py's Frames does (one ratio: output / render width).
"""
import os
import struct

import numpy as np


class ExpFrames:
    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = f.read(64)
        self.path = path
        self.size = os.path.getsize(path)
        if len(hdr) < 64 or hdr[:4] != b"EXP1":  # (cut short as the game exited: complete() says no)
            self.end, self.lr = 1 << 62, {}
            return
        _, self.ow, self.oh, rw, rh = struct.unpack_from("<5I", hdr, 4)
        jit = struct.unpack_from("<2f", hdr, 24)
        self.proj = struct.unpack_from("<4f", hdr, 32)
        self.ratio = round(self.ow / rw)
        self.ref_off = 64
        oc = 64 + self.ow * self.oh * 4
        n = rw * rh
        self.lr = {self.ratio: (rw, rh, jit, oc, oc + n * 4, oc + n * 8)}
        self.end = oc + n * 12

    def complete(self):
        return self.size >= self.end

    def _rows(self, f, off, width, y, h, x, w, dtype, ch):
        bpp = np.dtype(dtype).itemsize * ch
        f.seek(off + y * width * bpp)
        return np.frombuffer(f.read(h * width * bpp), dtype).reshape(h, width, ch)[:, x:x + w]

    def ref(self, x, y, w, h):
        """The teacher's output, alpha: the interface mask (the render size's, nearest)."""
        r = self.ratio
        rw, rh, _, oc, _, _ = self.lr[r]
        with open(self.path, "rb") as f:
            out = np.array(self._rows(f, self.ref_off, self.ow, y, h, x, w, np.uint8, 4))
            m = self._rows(f, oc, rw, y // r, (h + r - 1) // r + 1, x // r, (w + r - 1) // r + 1, np.uint8, 4)[..., 3]
        yy = (np.arange(h) + y) // r - y // r
        xx = (np.arange(w) + x) // r - x // r
        out[..., 3] = m[yy][:, xx]
        return out

    def low(self, r, x, y, w, h):
        lw, lh, jit, oc, od, om = self.lr[r]
        with open(self.path, "rb") as f:
            c = self._rows(f, oc, lw, y, h, x, w, np.uint8, 4)
            d = self._rows(f, od, lw, y, h, x, w, np.float32, 1)[..., 0]
            m = self._rows(f, om, lw, y, h, x, w, np.float16, 2)
        return c, d, m, jit
