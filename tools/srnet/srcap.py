"""Reader of the runtime's training captures (LARECOMP_SRCAP, runtime/nvn/vulkan/exec.c srcap_run).

One file per real frame, f<present>.bin: a 64-byte header ("SRC1", present, output width and height, the projection
P00 P11 A B, then for the ratios 2 and 3 the low-resolution size and the jitter in render pixels), then the planes:
the reference RGBA8 at the output size (alpha: the interface mask's share), and per ratio the colour RGBA8 (alpha: the
mask), the depth R32F and the motion vectors RG16F (uv now - uv before).
"""
import os
import struct

import numpy as np


def read_frame(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != b"SRC1":
        raise ValueError(f"{path}: not a capture")
    present, ow, oh = struct.unpack_from("<3I", buf, 4)
    proj = struct.unpack_from("<4f", buf, 16)
    off = 64
    ref = np.frombuffer(buf, np.uint8, ow * oh * 4, off).reshape(oh, ow, 4)
    off += ow * oh * 4
    fr = {"present": present, "proj": proj, "ref": ref}
    for r in range(2):
        lw, lh = struct.unpack_from("<2I", buf, 32 + 16 * r)
        jx, jy = struct.unpack_from("<2f", buf, 40 + 16 * r)
        n = lw * lh
        col = np.frombuffer(buf, np.uint8, n * 4, off).reshape(lh, lw, 4)
        off += n * 4
        dep = np.frombuffer(buf, np.float32, n, off).reshape(lh, lw)
        off += n * 4
        mv = np.frombuffer(buf, np.float16, n * 2, off).reshape(lh, lw, 2)
        off += n * 4
        fr[2 + r] = {"color": col, "depth": dep, "mv": mv, "jitter": (jx, jy)}
    return fr


def list_frames(folder):
    """The captures in a folder, sorted, split into runs of consecutive presents (each a sequence)."""
    names = sorted(n for n in os.listdir(folder) if n.startswith("f") and n.endswith(".bin"))
    seqs, cur, last = [], [], None
    for n in names:
        p = int(n[1:-4])
        if last is not None and p != last + 1:
            seqs.append(cur)
            cur = []
        cur.append(os.path.join(folder, n))
        last = p
    if cur:
        seqs.append(cur)
    return seqs
