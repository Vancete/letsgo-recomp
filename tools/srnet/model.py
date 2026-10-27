"""DittoSR Experiment 2's upscale (runtime/nvn/vulkan/shaders.c gsr2_convert.frag, x2_upscale.frag, x2_final.frag)
in PyTorch, differentiable, with the tiny network's per-pixel corrections in it.

The network runs at the render size and only steers the hard parts of the classic pipeline (zero output: Experiment
2 as it is): how far the history is trusted (the blend weight), how wide its clip box is, how narrow and how
edge-stretched the reconstruction kernel is, and how strong the final line pass is. It is recurrent: two channels of
its own state, carried to the next frame along the motion vectors.

Layout: tensors B x C x H x W. Coordinates as the shaders': a render pixel p's sample sits at p + 0.5 - jitter (render
pixels), uv = (output pixel + 0.5) / output size, the motion vectors are uv now - uv before (uv units), and the
history is the previous upscale (before the line pass).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Experiment 2's defaults (exec.c gsr2_run): base weight at rest, the narrow estimate's confidence, the narrow kernel's
# width, the history clip width at rest, the weight by confidence (b.x, b.y), reaction, edge; the line pass's strengths
# (dark, light: 3, 1 x 2 x the 0.5 slider; below 0.4).
ALPHA, CONF, WIDTH, REST, AFAR, ANEAR, REACT, EDGE = 0.05, 0.3, 1.4, 1.5, 0.6, 1.4, 1.0, 1.0
LINE_DARK, LINE_LIGHT, LINE_BELOW = 3.0, 1.0, 0.4
LUMA = (0.25, 0.5, 0.25)


def ycc(c):
    r, g, b = c[:, 0:1], c[:, 1:2], c[:, 2:3]
    return torch.cat([0.25 * r + 0.5 * g + 0.25 * b, 0.5 * (r - b), 0.5 * g - 0.25 * (r + b)], 1)


def rgb(y):
    a, b, c = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    return torch.cat([a + b - c, a + c, a - b - c], 1)


def sample(img, uv):
    """Bilinear, clamped to the edge (the runtime's linear sampler). uv: B x H x W x 2 in 0..1."""
    return F.grid_sample(img, uv * 2 - 1, mode="bilinear", padding_mode="border", align_corners=False)


def catmull_rom(img, uv):
    """x2_upscale.frag's hist(): Catmull-Rom from 5 bilinear taps (the corners left out), clamped to 0..1."""
    h, w = img.shape[-2:]
    sz = torch.tensor([w, h], device=img.device, dtype=img.dtype)
    p = uv * sz
    c1 = torch.floor(p - 0.5) + 0.5
    f = p - c1
    w0 = f * (-0.5 + f * (1.0 - 0.5 * f))
    w1 = 1.0 + f * f * (-2.5 + 1.5 * f)
    w2 = f * (0.5 + f * (2.0 - 1.5 * f))
    w3 = f * f * (-0.5 + 0.5 * f)
    w12 = w1 + w2
    q0, q3, q12 = (c1 - 1.0) / sz, (c1 + 2.0) / sz, (c1 + w2 / w12) / sz

    def at(x, y):
        return sample(img, torch.stack([x, y], -1))

    def wt(a):
        return a.unsqueeze(1)

    r = (at(q12[..., 0], q0[..., 1]) * wt(w12[..., 0] * w0[..., 1])
         + at(q0[..., 0], q12[..., 1]) * wt(w0[..., 0] * w12[..., 1])
         + at(q12[..., 0], q12[..., 1]) * wt(w12[..., 0] * w12[..., 1])
         + at(q3[..., 0], q12[..., 1]) * wt(w3[..., 0] * w12[..., 1])
         + at(q12[..., 0], q3[..., 1]) * wt(w12[..., 0] * w3[..., 1]))
    ws = (w12[..., 0] * w0[..., 1] + w0[..., 0] * w12[..., 1] + w12[..., 0] * w12[..., 1]
          + w3[..., 0] * w12[..., 1] + w12[..., 0] * w3[..., 1])
    return (r / wt(ws)).clamp(0, 1)


def depth_clip(depth, tan_fov, diag):
    """gsr2_convert.frag's depth clip (z): how far a pixel's depth stands out from its 4x4 neighbourhood.
    depth: B x 1 x h x w (1: far); tan_fov: B; diag: the length of the render size the runtime would use."""
    d = F.pad(depth, (1, 2, 1, 2), mode="replicate")
    h, w = depth.shape[-2:]

    def t(dx, dy):  # the texel at p + (dx, dy)
        return d[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    # textureGather at the corner p (texels p-1, p): x = (i0, j1), y = (i1, j1), z = (i1, j0), w = (i0, j0)
    def gather(ox, oy):
        i0, j0 = ox - 1, oy - 1
        return t(i0, j0 + 1), t(i0 + 1, j0 + 1), t(i0 + 1, j0), t(i0, j0)

    bl, br, tl, tr = gather(0, 0), gather(2, 0), gather(0, 2), gather(2, 2)
    maxc = torch.minimum(torch.minimum(bl[2], br[3]), torch.minimum(tl[1], tr[0]))

    def min4(g):
        return torch.minimum(torch.minimum(g[0], g[1]), torch.minimum(g[2], g[3]))

    sep = (1.37e-05 * tan_fov * diag).view(-1, 1, 1, 1) * (1.0 - maxc)
    wd = sum((sep / ((maxc - min4(g)).abs() + 1.19e-07)).clamp(0, 1) for g in (bl, br, tl, tr))
    clip = (1.0 - wd * 0.25).clamp(0, 1)
    return torch.where(maxc < 1.0 - 1.0e-05, clip, torch.zeros_like(clip))


def smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class TinyNet(nn.Module):
    """The network: 10 inputs at the render size -> 5 corrections + 2 state channels (+ res: a detail residual, the
    output's r x r pixels per render pixel - LOCAL research, the teacher's students). ~3.9k parameters at hid 16.
    sep: the second layer separable - a 3x3 per channel (c2d), then a 1x1 across them (c2p): 9 + hid instead of 9 x hid
    products per output channel."""
    IN, HID, OUT, STATE = 10, 16, 5, 2

    def __init__(self, hid=16, res=0, sep=False):
        super().__init__()
        self.c1 = nn.Conv2d(self.IN, hid, 3, padding=1)
        if sep:
            self.c2d = nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False)
            self.c2p = nn.Conv2d(hid, hid, 1)
        else:
            self.c2 = nn.Conv2d(hid, hid, 3, padding=1)
        self.c3 = nn.Conv2d(hid, self.OUT + self.STATE + res, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2p(self.c2d(x)) if hasattr(self, "c2p") else self.c2(x))
        return self.c3(x)


class Upscaler(nn.Module):
    """opts (DittoSR's ghosting and settling, 1 on as the runtime's - built in there; 0 off): greact (the reaction to a
    large change of the image), dilate (the nearest motion of the 3x3), vrej (the rejection by a changed motion), minmax
    (the box within the samples' range in motion); drej (the rejection by a changed depth: tried, not taken - off)."""
    def __init__(self, use_net=True, opts=None, hid=16, res_ratio=0, sep=False):
        super().__init__()
        self.res_ratio = res_ratio  # (a detail residual added to the final image: res_ratio^2 x 3 channels, pixel-shuffled)
        self.net = TinyNet(hid, 3 * res_ratio * res_ratio, sep) if use_net else None
        self.opts = {"greact": 1.0, "drej": 0.0, "dilate": 1.0, "vrej": 1.0, "minmax": 1.0, **(opts or {})}
        self.prev_mda = None

    def features(self, color, clip, mv, hist, state, os_):
        """The network's inputs at the render size: the colour (YCoCg), the depth clip, how far the pixel moves, the
        history reprojected to it (as a difference from the colour), the state carried along the motion."""
        b, _, h, w = color.shape
        ys = (torch.arange(h, device=color.device, dtype=color.dtype) + 0.5) / h
        xs = (torch.arange(w, device=color.device, dtype=color.dtype) + 0.5) / w
        uv = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).expand(b, h, w, 2)
        puv = uv - mv.permute(0, 2, 3, 1)
        yc = ycc(color)
        hy = ycc(sample(hist, puv)) - yc if hist is not None else torch.zeros_like(yc)
        st = sample(state, puv) if state is not None else torch.zeros(b, 2, h, w, device=color.device, dtype=color.dtype)
        mlen = torch.log1p((mv * torch.tensor([os_[0], os_[1]], device=mv.device, dtype=mv.dtype).view(1, 2, 1, 1)).norm(dim=1, keepdim=True))
        return torch.cat([yc, clip, mlen, hy, st], 1)

    def forward(self, color, depth, mv, jit, hist, state, out_size, tan_fov, diag, origin=None, full_size=None):
        """One frame. color B x 3 x h x w (0..1), depth B x 1 x h x w, mv B x 2 x h x w (uv units of this crop), jit B x 2
        (render pixels), hist: the previous upscale (B x 3 x H x W) or None (a reset), state: the network's or None.
        origin / full_size: this crop's place in the whole output image (the off-screen test as the runtime's).
        Returns (the upscale - the next history, the final image after the line pass, the next state)."""
        B, _, h, w = color.shape
        W, H = out_size
        dev, dt = color.device, color.dtype
        ratio = W / w
        o_ = self.opts
        clip = depth_clip(depth, tan_fov, diag)
        if o_["dilate"] > 0.5:  # (gsr2_convert.frag e.x: the motion of the nearest of the 3x3)
            dp = F.pad(depth, (1, 1, 1, 1), mode="replicate")
            mp = F.pad(mv, (1, 1, 1, 1), mode="replicate")
            ds = torch.stack([dp[:, 0, y:y + h, x:x + w] for y in range(3) for x in range(3)], 1)
            ms = torch.stack([mp[:, :, y:y + h, x:x + w] for y in range(3) for x in range(3)], 1)
            # (the shader takes the first minimum in its order: argmin does too)
            k = ds.argmin(1, keepdim=True)
            mv = torch.gather(ms, 1, k.unsqueeze(2).expand(-1, -1, 2, -1, -1)).squeeze(1)
        mda = torch.cat([2.0 * mv, clip, torch.log2((1.0 - depth).clamp_min(1e-7))], 1)
        pmda = self.prev_mda if hist is not None else None
        self.prev_mda = mda.detach()
        corr = None
        new_state = None
        if self.net is not None:
            o = self.net(self.features(color, clip, mv, hist, state, (W, H)))
            corr = 2.0 * torch.tanh(0.5 * o[:, :TinyNet.OUT])
            new_state = torch.tanh(o[:, TinyNet.OUT:TinyNet.OUT + TinyNet.STATE])
            resid = F.pixel_shuffle(o[:, TinyNet.OUT + TinyNet.STATE:], self.res_ratio) if self.res_ratio else None
        # the output pixels' uv and render coordinates
        ys = (torch.arange(H, device=dev, dtype=dt) + 0.5) / H
        xs = (torch.arange(W, device=dev, dtype=dt) + 0.5) / W
        uv = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).expand(B, H, W, 2)
        rs = torch.tensor([w, h], device=dev, dtype=dt)
        q = uv * rs
        j = jit.view(B, 1, 1, 2)
        b0 = torch.floor(q + j)
        ks = max(ratio, 1.0) / WIDTH
        # the 3x3 render samples around the pixel
        flat = color.flatten(2)
        cc, dd = [], []
        for i in range(9):
            off = torch.tensor([i % 3 - 1, i // 3 - 1], device=dev, dtype=dt)
            p = b0 + off
            pi = p.long()
            px = pi[..., 0].clamp(0, w - 1)
            py = pi[..., 1].clamp(0, h - 1)
            idx = (py * w + px).view(B, 1, -1).expand(B, 3, -1)
            cc.append(torch.gather(flat, 2, idx).view(B, 3, H, W))
            dd.append((p + 0.5 - j - q).permute(0, 3, 1, 2))
        l = [(c * torch.tensor(LUMA, device=dev, dtype=dt).view(1, 3, 1, 1)).sum(1, keepdim=True) for c in cc]
        gx = (l[2] + 2 * l[5] + l[8]) - (l[0] + 2 * l[3] + l[6])
        gy = (l[6] + 2 * l[7] + l[8]) - (l[0] + 2 * l[1] + l[2])
        lst = torch.cat(l, 1)
        lmn, lmx = lst.min(1, keepdim=True)[0], lst.max(1, keepdim=True)[0]
        g = torch.sqrt(gx * gx + gy * gy + 1e-12)
        es = (g / (4.0 * (lmx - lmn) + 1e-3)).clamp(0, 1)
        nx = torch.where(g > 1e-5, gx / g, torch.ones_like(g))
        ny = torch.where(g > 1e-5, gy / g, torch.zeros_like(g))
        along = 1.0 - 0.75 * EDGE * es
        if corr is not None:  # (the corrections at the output pixel: bilinear from the render size)
            cu = sample(corr, uv)
            ks_c, al_c, gam_c, a_c, line_c = (cu[:, k:k + 1] for k in range(5))
            ks = ks * torch.exp(0.5 * ks_c)
            along = (along * torch.exp(al_c)).clamp(0.05, 4.0)
        sg = torch.zeros(B, 3, H, W, device=dev, dtype=dt)
        ss = torch.zeros_like(sg)
        m1 = torch.zeros_like(sg)
        m2 = torch.zeros_like(sg)
        ymn = torch.full_like(sg, 1e9)
        ymx = torch.full_like(sg, -1e9)
        wg = torch.zeros(B, 1, H, W, device=dev, dtype=dt)
        wsum = torch.zeros_like(wg)
        for i in range(9):
            d = dd[i]
            dn = d[:, 0:1] * nx + d[:, 1:2] * ny
            dtn = -d[:, 0:1] * ny + d[:, 1:2] * nx
            wn = torch.exp(-2.26 * (dn * dn + dtn * dtn * along) * ks * ks)
            w2 = torch.exp(-1.4 * (d * d).sum(1, keepdim=True))
            sg = sg + cc[i] * wn
            wg = wg + wn
            ss = ss + cc[i] * w2
            wsum = wsum + w2
            yc = ycc(cc[i])
            m1 = m1 + yc
            m2 = m2 + yc * yc
            ymn = torch.minimum(ymn, yc)
            ymx = torch.maximum(ymx, yc)
        m1 = m1 / 9.0
        sig = torch.sqrt((m2 / 9.0 - m1 * m1).clamp_min(0) + 1e-12)
        cg = sg / wg.clamp_min(1e-6)
        cs2 = ss / wsum.clamp_min(1e-6)
        conf = wg.clamp(0, 1)
        cur = cs2 + (cg - cs2) * smoothstep(0.05, CONF, conf)
        if hist is None:
            out = cs2
        else:
            m = sample(mda, uv)
            mvo = 0.5 * m[:, 0:2]
            osz = torch.tensor([W, H], device=dev, dtype=dt).view(1, 2, 1, 1)
            puv = uv - mvo.permute(0, 2, 3, 1)
            mlen = (mvo * torch.tensor([W, H], device=dev, dtype=dt).view(1, 2, 1, 1)).norm(dim=1, keepdim=True)
            mz = m[:, 2:3] * (mlen * 0.5).clamp(0, 1)
            # (off the whole image: the runtime's test, in its coordinates)
            if origin is not None:
                fs = torch.tensor(full_size, device=dev, dtype=dt)
                guv = (uv * torch.tensor([W, H], device=dev, dtype=dt) + origin.view(B, 1, 1, 2)) / fs
                gp = guv - mvo.permute(0, 2, 3, 1) * torch.tensor([W, H], device=dev, dtype=dt) / fs
            else:
                gp = puv
            off = ((gp < 0) | (gp > 1)).any(-1).unsqueeze(1)
            hy = ycc(catmull_rom(hist, puv))
            # (x2_upscale.frag f.y / f.z: the previous frame's depth / motion where the pixel was)
            rj = torch.zeros_like(mz)
            if pmda is not None and (o_["drej"] > 0 or o_["vrej"] > 0):
                pm = sample(pmda, puv)
                rj = o_["drej"] * smoothstep(0.1, 0.3, (m[:, 3:4] - pm[:, 3:4]).abs())
                rj = torch.maximum(rj, o_["vrej"] * smoothstep(0.5, 2.0, (0.5 * (m[:, 0:2] - pm[:, 0:2]) * osz).norm(dim=1, keepdim=True)))
            gch = 0.0
            if o_["greact"] > 0:
                gch = o_["greact"] * smoothstep(0.2, 0.5, global_change(hist, mda, color))
            gam = (REST + (0.9 - REST) * (mlen * 0.25).clamp(0, 1)) * (1.0 - 0.5 * mz)
            if corr is not None:
                gam = gam * torch.exp(gam_c)
            gam = gam * (1.0 - 0.6 * rj)
            blo, bhi = m1 - gam * sig, m1 + gam * sig
            mm = o_["minmax"] * (mlen * 0.5).clamp(0, 1)
            blo = blo + (torch.maximum(blo, ymn) - blo) * mm
            bhi = bhi + (torch.minimum(bhi, ymx) - bhi) * mm
            bc, ext3 = 0.5 * (blo + bhi), 0.5 * (bhi - blo) + 1.0 / 255.0
            dirv = hy - bc
            t = (dirv.abs() / ext3).max(1, keepdim=True)[0]
            hy = torch.where(t > 1.0, bc + dirv / t.clamp_min(1.0), hy)
            base = ALPHA + (0.3 - ALPHA) * (mlen * 0.125).clamp(0, 1)
            a = base * (AFAR + (ANEAR - AFAR) * smoothstep(0.05, 0.8, conf))
            a = a + (0.6 - a) * mz
            tgt = 0.5 + (0.95 - 0.5) * gch
            a = a + (tgt - a) * ((t - 1.0) * REACT).clamp(0, 1)
            a = a + (torch.maximum(a, torch.full_like(a, 0.5)) - a) * gch
            a = a + (torch.maximum(a, torch.full_like(a, 0.6)) - a) * rj
            a = a.clamp(0, 1)
            if corr is not None:
                ac = a.clamp(1e-3, 1 - 1e-3)
                a = torch.sigmoid(torch.log(ac / (1 - ac)) + a_c)
            out = rgb(hy) + (cur - rgb(hy)) * a
            out = torch.where(off, cs2, out)
        out = out.clamp(0, 1)
        # the line pass (x2_final.frag)
        s = torch.exp(line_c) if corr is not None else 1.0
        final = line_pass(out, s)
        if corr is not None and self.res_ratio:
            final = (final + resid).clamp(0, 1)
        return out, final, new_state


def global_change(hist, mda, color):
    """x2_change.frag: the share of a 16 x 16 grid of render pixels whose colour differs from the history where the
    point was by over 0.1 in luma (B x 1 x 1 x 1)."""
    B, _, h, w = color.shape
    g = (torch.arange(16, device=color.device, dtype=color.dtype) + 0.5) / 16
    qx, qy = (g * w).long(), (g * h).long()
    c = color[:, :, qy][:, :, :, qx]
    mv = mda[:, 0:2, qy][:, :, :, qx]
    gy, gx = torch.meshgrid((qy.to(color.dtype) + 0.5) / h, (qx.to(color.dtype) + 0.5) / w, indexing="ij")
    puv = torch.stack([gx, gy], -1).expand(B, 16, 16, 2) - 0.5 * mv.permute(0, 2, 3, 1)
    valid = ((puv >= 0) & (puv <= 1)).all(-1).unsqueeze(1).to(color.dtype)
    lw = torch.tensor(LUMA, device=color.device, dtype=color.dtype).view(1, 3, 1, 1)
    d = ((c - sample(hist, puv)) * lw).sum(1, keepdim=True).abs()
    ch = (d > 0.1).to(color.dtype) * valid
    return (ch.sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1))


def line_pass(x, s=1.0):
    """x2_final.frag without the interface: an unsharp mask, dark lines LINE_DARK x s, light ones LINE_LIGHT x s,
    clamped to the 3x3's range (below its minimum by LINE_BELOW x the range)."""
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    H, W = x.shape[-2:]
    t = [p[:, :, dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]
    a, b, c, d, e, f, g, h, i = t
    blur = (4 * e + 2 * (b + d + f + h) + (a + c + g + i)) / 16
    det = e - blur
    lum = (det * torch.tensor(LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
    k = torch.where(lum < 0, LINE_DARK * s, LINE_LIGHT * s) if torch.is_tensor(s) else torch.where(lum < 0, torch.full_like(lum, LINE_DARK * s), torch.full_like(lum, LINE_LIGHT * s))
    r = e + k * det
    st = torch.stack(t, 0)
    mn, mx = st.min(0)[0], st.max(0)[0]
    return torch.maximum(torch.minimum(r, mx), (mn - LINE_BELOW * (mx - mn)).clamp_min(0))
