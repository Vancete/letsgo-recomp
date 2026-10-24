"""catch_button.py: the battle menu with a fifth button, CATCH (wild battles, docs/MODO-BATALHA.md stage 6), made from
the game's own files - the reference for runtime/hle/catch_button.c, which does the same at run time.

  catch_button.py <romfs> <out dir> [<art dir>]

<romfs>/bin/app/battle/bin/{battle_command_00*.arc, layout_setting_battle_command_00.bin, uikit_setting_...bin} ->
<out dir>/bin/app/battle/bin/... (the same paths). Art: catch_button_art.py's PNGs (made into <out dir>/art if not given).

What changes (the menu's buttons: docs/MODO-BATALHA.md, stage 6):
- each language archive: the BACK button's textures (pattern frame 4: only two-player battles show it) are redrawn as
  CATCH, same size and format; blyt/battle_command_button_catch.bflyt, a copy of the button's part layout that shows
  those textures and red colours without the pattern animation; in blyt/battle_command_00.bflyt the panes N_command_04 /
  L_command_04 (the part) after N_command_03 at RUN's place, the four buttons one step (136) to the left (the game
  places them again from B_bounding_00: five end where four did);
- layout_setting: the new panes (N_command_04, L_command_04 and its parts' panes) and L_command_04's animations (all of
  L_command_03's but the pattern), new hashes derived from the old ones;
- uikit_setting: a fifth button (a copy of RUN's, its hashes mapped the same way) and its cell in the navigation grid
  (column 4; the grid's columns 4 -> 5).
FlatBuffers are only appended to (new tables and vectors at the end, the root's fields pointed at them): offsets point
forward, so the vectors that grow are copied whole.
"""
import os
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIXES = ['', '_eng', '_fre', '_ger', '_ita', '_spa', '_kor', '_sch', '_tch']
STEP = float(os.environ.get("CATCH_STEP", "136"))  # the buttons' spacing in the layout


def fnv64(b):
    h = 0xcbf29ce484222645
    for c in b:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def derive(h):
    """A new hash for the copy of something whose hash is h (the game only compares them)."""
    return fnv64(b'%016x:catch' % h)


# ---------------------------------------------------------------- SARC
def sarc_read(d):
    dataoff = struct.unpack_from('<I', d, 0xc)[0]
    n = struct.unpack_from('<H', d, 0x1a)[0]
    key = struct.unpack_from('<I', d, 0x1c)[0]
    nodes = [struct.unpack_from('<IIII', d, 0x20 + i * 16) for i in range(n)]
    names = d[0x20 + n * 16 + 8:dataoff]
    files = []
    for h, attr, s, e in nodes:
        o = (attr & 0xffffff) * 4
        nm = names[o:names.index(b'\0', o)].decode()
        files.append([nm, d[dataoff + s:dataoff + e]])
    return files, key


def sarc_hash(name, key):
    h = 0
    for c in name.encode():
        h = (h * key + c) & 0xFFFFFFFF
    return h


def sarc_write(files, key=0x65):
    def align(name):
        return 0x1000 if name.endswith(('.bntx', '.bnsh')) else 0x10
    names = b''
    ent = []
    for nm, data in files:
        ent.append((sarc_hash(nm, key), len(names) // 4, nm, data))
        names += nm.encode() + b'\0'
        names += b'\0' * (-len(names) % 4)
    ent.sort(key=lambda e: e[0])
    hdr_len = 0x14 + 0xc + 16 * len(ent) + 8 + len(names)
    dataoff = (hdr_len + 0xfff) & ~0xfff
    blob = bytearray()
    spans = {}
    for nm, data in files:  # (the data in the original order)
        pad = -len(blob) % align(nm)
        blob += b'\0' * pad
        spans[nm] = (len(blob), len(blob) + len(data))
        blob += data
    out = bytearray(b'SARC' + struct.pack('<HHIIHH', 0x14, 0xFEFF, 0, dataoff, 0x100, 0))
    out += b'SFAT' + struct.pack('<HHI', 0xc, len(ent), key)
    for h, no, nm, _ in ent:
        s, e = spans[nm]
        out += struct.pack('<IIII', h, 0x01000000 | no, s, e)
    out += b'SFNT' + struct.pack('<HH', 8, 0) + names
    out += b'\0' * (dataoff - len(out))
    out += blob
    struct.pack_into('<I', out, 8, len(out))
    return bytes(out)


# ---------------------------------------------------------------- textures: Tegra block linear + BCn
def _gob(x, y, wg, bh):
    off = ((y // (8 * bh)) * 512 * bh * wg + (x // 64) * 512 * bh + ((y % (8 * bh)) // 8) * 512)
    xi, yi = x % 64, y % 8
    return off + (xi // 32) * 256 + (yi // 2) * 64 + ((xi % 32) // 16) * 32 + (yi % 2) * 16 + (xi % 16)


def swizzle(lin, ew, eh, bpe, blh_log2, size):
    bh = 1 << blh_log2
    wg = (ew * bpe + 63) // 64
    out = bytearray(size)
    for y in range(eh):
        for x in range(ew):
            for b in range(bpe):
                out[_gob(x * bpe + b, y, wg, bh)] = lin[(y * ew + x) * bpe + b]
    return bytes(out)


def blocks(a):
    """(h, w[, c]) -> 4x4 blocks, row-major, padded by edge pixels."""
    h, w = a.shape[:2]
    H, W = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    a = np.pad(a, ((0, H - h), (0, W - w)) + ((0, 0),) * (a.ndim - 2), mode='edge')
    return [a[y:y + 4, x:x + 4] for y in range(0, H, 4) for x in range(0, W, 4)], W // 4, H // 4


def bc4_block(v):
    v = v.astype(np.float64).ravel()
    hi, lo = v.max(), v.min()
    if hi == lo:
        return struct.pack('<BB', int(hi), int(lo)) + b'\0' * 6
    e0, e1 = int(round(hi)), int(round(lo))
    pal = [e0, e1] + [((7 - i) * e0 + i * e1) / 7 for i in range(1, 7)]
    idx = [min(range(8), key=lambda k: abs(pal[k] - x)) for x in v]
    bits = 0
    for i, k in enumerate(idx):
        bits |= k << (3 * i)
    return struct.pack('<BB', e0, e1) + bits.to_bytes(6, 'little')


def rgb565(c):
    r, g, b = (int(round(x)) for x in c)
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def unpack565(v):
    return np.array([((v >> 11) & 31) * 255 / 31, ((v >> 5) & 63) * 255 / 63, (v & 31) * 255 / 31])


def bc1_block(rgb, weight):
    """4-colour BC1 (c0 > c1): endpoints along the block's main axis (pixels weighted by alpha)."""
    p = rgb.reshape(16, 3).astype(np.float64)
    w = weight.ravel().astype(np.float64) + 1e-3
    mean = (p * w[:, None]).sum(0) / w.sum()
    cov = ((p - mean) * w[:, None]).T @ (p - mean)
    ax = np.linalg.eigh(cov)[1][:, -1]
    t = (p - mean) @ ax
    c0, c1 = mean + ax * t.max(), mean + ax * t.min()
    v0, v1 = rgb565(np.clip(c0, 0, 255)), rgb565(np.clip(c1, 0, 255))
    if v0 < v1:
        v0, v1 = v1, v0
    if v0 == v1:
        return struct.pack('<HHI', v0, v1, 0)
    a, b = unpack565(v0), unpack565(v1)
    pal = [a, b, (2 * a + b) / 3, (a + 2 * b) / 3]
    bits = 0
    for i, x in enumerate(p):
        k = min(range(4), key=lambda j: ((pal[j] - x) ** 2).sum())
        bits |= k << (2 * i)
    return struct.pack('<HHI', v0, v1, bits)


def encode(img, fmt):
    """img: numpy (h, w) / (h, w, 2) / (h, w, 4) uint8 -> linear BCn blocks. fmt: 0x1c BC3, 0x1d BC4, 0x1e BC5."""
    bl, bw, bhh = blocks(img)
    out = bytearray()
    for b in bl:
        if fmt == 0x1d:
            out += bc4_block(b)
        elif fmt == 0x1e:
            out += bc4_block(b[..., 0]) + bc4_block(b[..., 1])
        elif fmt == 0x1c:
            out += bc4_block(b[..., 3]) + bc1_block(b[..., :3], b[..., 3])
        else:
            raise ValueError(hex(fmt))
    return bytes(out), bw, bhh


def bntx_textures(d):
    count = struct.unpack_from('<I', d, 0x24)[0]
    info = struct.unpack_from('<Q', d, 0x28)[0]
    out = {}
    for i in range(count):
        p = struct.unpack_from('<Q', d, info + 8 * i)[0]
        fmt = struct.unpack_from('<I', d, p + 0x1c)[0]
        w, h = struct.unpack_from('<II', d, p + 0x24)
        blh = struct.unpack_from('<I', d, p + 0x34)[0]
        size = struct.unpack_from('<I', d, p + 0x50)[0]
        name_off = struct.unpack_from('<Q', d, p + 0x60)[0]
        n = struct.unpack_from('<H', d, name_off)[0]
        name = d[name_off + 2:name_off + 2 + n].decode()
        data = struct.unpack_from('<Q', d, struct.unpack_from('<Q', d, p + 0x70)[0])[0]
        out[name.split('^')[0].split('+')[0]] = dict(fmt=fmt >> 8, w=w, h=h, blh=blh, size=size, data=data)
    return out


def bntx_replace(d, images):
    """images: {texture name (without the ^ suffix): numpy image}; same size and format as the texture's."""
    d = bytearray(d)
    tex = bntx_textures(d)
    for name, img in images.items():
        t = tex[name]
        assert img.shape[:2] == (t['h'], t['w']), (name, img.shape, t)
        lin, bw, bh = encode(img, t['fmt'])
        bpe = 8 if t['fmt'] == 0x1d else 16
        d[t['data']:t['data'] + t['size']] = swizzle(lin, bw, bh, bpe, t['blh'], t['size'])
    return bytes(d)


# ---------------------------------------------------------------- BFLYT
def lyt_sections(d):
    hsz = struct.unpack_from('<H', d, 6)[0]
    n = struct.unpack_from('<H', d, 0x10)[0]
    o, out = hsz, []
    for _ in range(n):
        m, sz = struct.unpack_from('<4sI', d, o)
        out.append([m, bytearray(d[o:o + sz])])
        o += sz
    return d[:hsz], out


def lyt_join(head, secs):
    body = b''.join(bytes(s) for _, s in secs)
    head = bytearray(head)
    struct.pack_into('<I', head, 0xc, len(head) + len(body))
    struct.pack_into('<H', head, 0x10, len(secs))
    return bytes(head) + body


def pane_name(sec):
    return bytes(sec[0xc:0x24]).split(b'\0')[0].decode()


def set_name(sec, name):
    sec[0xc:0x24] = name.encode().ljust(0x18, b'\0')


def part_layout(d):
    """battle_command_button_00.bflyt -> the CATCH copy: the BACK textures by default, red colours."""
    head, secs = lyt_sections(d)
    swap = {'battle_command_battle_off_00': 'battle_command_back_off_00',
            'battle_button_command_icon_00': 'battle_button_command_icon_08',
            'grpfont_battle_battleButton_text_00': 'grpfont_battle_backButton_text_00',
            'grpfont_battle_battleButton_text_01': 'grpfont_battle_backButton_text_01'}
    for i, (m, s) in enumerate(secs):
        if m == b'txl1':
            n = struct.unpack_from('<H', s, 8)[0]
            offs = [struct.unpack_from('<I', s, 0xc + 4 * k)[0] for k in range(n)]
            names = [bytes(s[0xc + o:]).split(b'\0')[0].decode() for o in offs]
            new = []
            for nm in names:
                base, sep, suffix = nm.partition('^')
                new.append(swap.get(base, base) + sep + suffix)
            body = b''
            table = b''
            for nm in new:
                table += struct.pack('<I', 4 * n + len(body))
                body += nm.encode() + b'\0'
            sec = bytearray(b'txl1' + struct.pack('<IHH', 0, n, 0) + table + body)
            sec += b'\0' * (-len(sec) % 4)
            struct.pack_into('<I', sec, 4, len(sec))
            secs[i][1] = sec
        if m == b'mat1':
            colours = {  # material: (black, white) RGBA; the FIGHT ones are pink / orange
                'N_glare_00': (None, (255, 64, 64, 255)),
                'P_buttonBG_00': (None, (232, 24, 40, 255)),
                'P_buttonBG_01': (None, (255, 255, 255, 255)),
                'P_buttonBG_02': (None, (255, 112, 112, 255)),
                'P_buttonBG_03': (None, (200, 0, 24, 255)),
                'P_grptext_00': ((150, 0, 16, 0), None),  # (alpha 0, as the game's: a mod's label with alpha shows it)
            }
            for nm, (black, white) in colours.items():
                j = bytes(s).find(nm.encode() + b'\0')
                assert j > 0, nm
                if black:
                    s[j + 0x24:j + 0x28] = bytes(black)
                if white:
                    s[j + 0x28:j + 0x2c] = bytes(white)
    return lyt_join(head, secs)


def menu_layout(d):
    """battle_command_00.bflyt: N_command_04 / L_command_04 after N_command_03, the buttons one step left."""
    head, secs = lyt_sections(d)
    names = [pane_name(s) if m in (b'pan1', b'prt1', b'pic1', b'bnd1') else None for m, s in secs]
    i3 = names.index('N_command_03')
    assert [m for m, _ in secs[i3:i3 + 4]] == [b'pan1', b'pas1', b'prt1', b'pae1'], secs[i3:i3 + 4]
    new = [[m, bytearray(s)] for m, s in secs[i3:i3 + 4]]
    set_name(new[0][1], 'N_command_04')
    set_name(new[2][1], 'L_command_04')
    part = new[2][1]
    assert bytes(part[0x60:0x7c]).split(b'\0')[0] == b'battle_command_button_00'
    part[0x60:0x7c] = b'battle_command_button_catch'.ljust(0x1c, b'\0')
    x3 = struct.unpack_from('<f', secs[i3][1], 0x2c)[0]
    struct.pack_into('<f', new[0][1], 0x2c, x3)
    for k in range(4):
        s = secs[names.index('N_command_%02d' % k)][1]
        struct.pack_into('<f', s, 0x2c, struct.unpack_from('<f', s, 0x2c)[0] - STEP)
    # the game places the buttons from B_bounding_00's right edge (RUN at the edge - 66, the others every 136 to its
    # left: f_7000ab2b70's layout), and with five it starts one step further left by itself: the area stays, so CATCH
    # ends where RUN did
    secs[i3 + 4:i3 + 4] = new
    return lyt_join(head, secs)


# ---------------------------------------------------------------- FlatBuffers (append only)
class FB:
    def __init__(self, d):
        self.d = bytearray(d)

    def u32(self, o): return struct.unpack_from('<I', self.d, o)[0]
    def i32(self, o): return struct.unpack_from('<i', self.d, o)[0]
    def u16(self, o): return struct.unpack_from('<H', self.d, o)[0]
    def u64(self, o): return struct.unpack_from('<Q', self.d, o)[0]
    def ref(self, a): return a + self.u32(a)

    def vt(self, t): return t - self.i32(t)

    def fields(self, t):
        vt = self.vt(t)
        n = (self.u16(vt) - 4) // 2
        return {i: t + self.u16(vt + 4 + 2 * i) for i in range(n) if self.u16(vt + 4 + 2 * i)}

    def vec(self, a):
        t = self.ref(a)
        return [t + 4 + 4 * i + self.u32(t + 4 + 4 * i) for i in range(self.u32(t))]

    def string(self, a):
        t = self.ref(a)
        return self.d[t + 4:t + 4 + self.u32(t)].decode()

    def strings(self, a):
        return [self.d[e + 4:e + 4 + self.u32(e)].decode() for e in self.vec(a)]

    # appending
    def pad(self, n, mod=0):
        while len(self.d) % n != mod:
            self.d.append(0)

    def put_string(self, s):
        self.pad(4)
        at = len(self.d)
        b = s.encode()
        self.d += struct.pack('<I', len(b)) + b + b'\0'
        return at

    def put_vector(self, n):
        """Reserves a vector of n offsets; returns its address (fill with set_off)."""
        self.pad(4)
        at = len(self.d)
        self.d += struct.pack('<I', n) + b'\0' * (4 * n)
        return at

    def set_off(self, slot, target):
        assert target > slot, (slot, target)
        struct.pack_into('<I', self.d, slot, target - slot)

    def put_table(self, like, values):
        """A table with the vtable (and size) of table `like`; values: {field: bytes | ('str', s) | ('strs', [s..])}.
        Offset fields are written after the table."""
        vt = self.vt(like)
        size = self.u16(vt + 2)
        self.pad(8, like % 8)
        at = len(self.d)
        self.d += b'\0' * size
        struct.pack_into('<i', self.d, at, at - vt)
        f = self.fields(at)
        later = []
        for k, v in values.items():
            if isinstance(v, (bytes, bytearray)):
                self.d[f[k]:f[k] + len(v)] = v
            else:
                later.append((f[k], v))
        for slot, (kind, v) in later:
            if kind == 'str':
                self.set_off(slot, self.put_string(v))
            else:
                vec = self.put_vector(len(v))
                for i, s in enumerate(v):
                    self.set_off(vec + 4 + 4 * i, self.put_string(s))
                self.set_off(slot, vec)
        return at

    def table_values(self, t, kinds):
        """The values of table t's fields as put_table takes them. kinds: {field: 'str' | 'strs'}; others are raw bytes
        up to the next field (scalars)."""
        f = self.fields(t)
        order = sorted(f.items(), key=lambda kv: kv[1])
        size = self.u16(self.vt(t) + 2)
        out = {}
        for i, (k, a) in enumerate(order):
            if kinds.get(k) == 'str':
                out[k] = ('str', self.string(a))
            elif kinds.get(k) == 'strs':
                out[k] = ('strs', self.strings(a))
            else:
                end = order[i + 1][1] if i + 1 < len(order) else t + size
                width = 8 if end - a >= 8 else 4 if end - a >= 4 else end - a
                out[k] = bytes(self.d[a:a + width])
        return out

    def regrow(self, slot, tables, kinds):
        """The vector at field address `slot`, made again at the end with tables (list of (like, values))."""
        vec = self.put_vector(len(tables))
        for i, (like, vals) in enumerate(tables):
            self.set_off(vec + 4 + 4 * i, self.put_table(like, vals))
        self.set_off(slot, vec)


def q(v): return struct.pack('<Q', v)
def u(v): return struct.pack('<I', v)


def layout_setting(d):
    fb = FB(d)
    root = fb.fields(fb.u32(0))
    anim_kinds = {1: 'strs', 2: 'str'}
    pane_kinds = {1: 'str', 2: 'strs'}
    # animations: L_command_03's but the pattern, for L_command_04
    anims = [(e, fb.table_values(e, anim_kinds)) for e in fb.vec(root[2])]
    extra = []
    for e, v in anims:
        if 1 in v and v[1][1] == ['L_command_03'] and 'patern' not in v[2][1]:
            nv = dict(v)
            nv[1] = ('strs', ['L_command_04'])
            nv[3] = q(derive(struct.unpack('<Q', v[3])[0]))
            extra.append((e, nv))
    panes = [(e, fb.table_values(e, pane_kinds)) for e in fb.vec(root[3])]
    extra_p = []
    for e, v in panes:
        name = v[1][1]
        path = v[2][1] if 2 in v else None
        if name in ('N_command_03', 'L_command_03') or path == ['L_command_03']:
            nv = dict(v)
            if name in ('N_command_03', 'L_command_03'):
                nv[1] = ('str', name.replace('_03', '_04'))
            else:
                nv[2] = ('strs', ['L_command_04'])
            nv[3] = q(derive(struct.unpack('<Q', v[3])[0]))
            extra_p.append((e, nv))
    assert len(extra) == 6 and len(extra_p) == 20, (len(extra), len(extra_p))
    fb.regrow(root[2], anims + extra, anim_kinds)
    fb.regrow(root[3], panes + extra_p, pane_kinds)
    return bytes(fb.d)


def uikit_setting(d):
    fb = FB(d)
    root = fb.fields(fb.u32(0))
    buttons = [(e, fb.table_values(e, {})) for e in fb.vec(root[8])]
    run_e, run = buttons[3]
    catch = {k: q(derive(struct.unpack('<Q', v)[0])) for k, v in run.items()}
    nav = fb.vec(root[9])[0]
    navf = fb.fields(nav)
    cells = [(e, fb.table_values(e, {})) for e in fb.vec(navf[4])]
    like, cell = cells[3]
    assert struct.unpack('<Q', cell[0])[0] == struct.unpack('<Q', run[0])[0] and struct.unpack('<I', cell[1])[0] == 3
    new_cell = dict(cell)
    new_cell[0] = catch[0]
    new_cell[1] = u(4)
    assert fb.u32(navf[1]) == 4  # (the grid's columns)
    struct.pack_into('<I', fb.d, navf[1], 5)
    fb.regrow(root[8], buttons + [(run_e, catch)], {})
    fb.regrow(navf[4], cells + [(like, new_cell)], {})
    return bytes(fb.d), struct.unpack('<Q', catch[0])[0]


# ---------------------------------------------------------------- the art
def art(art_dir, suffix):
    def load(name, mode):
        return np.asarray(Image.open(os.path.join(art_dir, name)).convert(mode))
    lang = suffix or '_jpn'
    t00 = load(f'text_00{lang}.png', 'L')
    return {
        'battle_command_back_off_00': load('off.png', 'RGBA'),
        'battle_button_command_icon_08': load('icon.png', 'L'),
        'grpfont_battle_backButton_text_00': np.dstack([t00, t00]),
        'grpfont_battle_backButton_text_01': load(f'text_01{lang}.png', 'L'),
    }


def main():
    romfs, out = sys.argv[1], sys.argv[2]
    art_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(out, 'art')
    if not os.path.exists(os.path.join(art_dir, 'off.png')):
        subprocess.check_call([sys.executable, os.path.join(HERE, 'catch_button_art.py'), art_dir])
    src = os.path.join(romfs, 'bin/app/battle/bin')
    dst = os.path.join(out, 'bin/app/battle/bin')
    os.makedirs(dst, exist_ok=True)
    for sfx in SUFFIXES:
        name = f'battle_command_00{sfx}.arc'
        files, key = sarc_read(open(os.path.join(src, name), 'rb').read())
        byname = {n: i for i, (n, _) in enumerate(files)}
        files[byname['timg/__Combined.bntx']][1] = bntx_replace(files[byname['timg/__Combined.bntx']][1], art(art_dir, sfx))
        files[byname['blyt/battle_command_00.bflyt']][1] = menu_layout(files[byname['blyt/battle_command_00.bflyt']][1])
        files.append(['blyt/battle_command_button_catch.bflyt', part_layout(files[byname['blyt/battle_command_button_00.bflyt']][1])])
        open(os.path.join(dst, name), 'wb').write(sarc_write(files, key))
        print(name)
    for name, fn in (('layout_setting_battle_command_00.bin', layout_setting), ('uikit_setting_battle_command_00.bin', uikit_setting)):
        r = fn(open(os.path.join(src, name), 'rb').read())
        if isinstance(r, tuple):
            r, h = r
            print('catch button hash %#x' % h)
        open(os.path.join(dst, name), 'wb').write(r)
        print(name)


if __name__ == '__main__':
    main()
