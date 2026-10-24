"""catch_button_art.py: the artwork of the battle menu's CATCH button (wild battles, docs/MODO-BATALHA.md stage 6),
drawn from scratch in the style of the game's command buttons (none of the game's images is used).

  catch_button_art.py <out dir>     writes the PNGs (circle, icon mask, the two label textures per language) and a
                                    contact sheet, contact.png
  catch_button_art.py <out dir> --header <file.h>
                                    also the PNGs as C arrays, for runtime/hle/catch_button.c
                                    (runtime/hle/catch_button_art.h)

The textures replace the battle menu's BACK button (only shown in two-player battles) in the language archives, so they
keep its sizes: circle 100x100 RGBA (the button's face, icon included), icon mask 200x200 (the glow layer), label
160x128 in two variants - text_00 (the shadow: the text thickened, blurred and inverted) and text_01 (the text, white).
The game's labels: Latin caps ~27 px high around y 53..80; Asian ones big (~33 px, y 42..76) with the English word small
below (~11 px, y 86..97).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = 'C:/Windows/Fonts/'
LATIN = (FONTS + 'bahnschrift.ttf', b'Bold SemiCondensed')

# language archive suffix -> (label, font, variation, italic shear, small English word below)
LABELS = {
    '':     ('つかまえる', FONTS + 'YuGothB.ttc', None, 0.18, True),   # Japanese (the base archive)
    '_eng': ('CATCH', *LATIN, 0, False),
    '_fre': ('CAPTURE', *LATIN, 0, False),
    '_ger': ('Fangen', *LATIN, 0, False),
    '_ita': ('CATTURA', *LATIN, 0, False),
    '_spa': ('CAPTURAR', *LATIN, 0, False),
    '_kor': ('잡는다', FONTS + 'malgunbd.ttf', None, 0, True),
    '_sch': ('捕捉', FONTS + 'msyhbd.ttc', None, 0, True),
    '_tch': ('捕捉', FONTS + 'msjhbd.ttc', None, 0, True),
}

RED = np.array([246, 92, 104])
DEEP = np.array([222, 44, 66])


def font(path, size, var=None):
    f = ImageFont.truetype(path, size)
    if var:
        f.set_variation_by_name(var)
    return f


def text_mask(text, path, var, height, shear=0.0, max_w=138, tracking=0, ref=None):
    """The text drawn white on black, cropped, scaled so its ink is `height` px high (narrower if wider than max_w).
    ref: scale by that text's ink height instead (a capital: mixed-case words keep the capitals' height)."""
    big = 4
    f = font(path, height * big * 2, var)
    w = int(f.getlength(text) + height * big * 4 + tracking * big * len(text))
    im = Image.new('L', (w, height * big * 5), 0)
    d = ImageDraw.Draw(im)
    x = height * big
    for ch in text:
        d.text((x, height * big), ch, font=f, fill=255)
        x += f.getlength(ch) + tracking * big
    if shear:
        im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * im.size[1] / 2, 0, 1, 0), Image.BICUBIC)
    box = im.getbbox()
    im = im.crop(box)
    s = height / im.size[1]
    if ref:
        r = Image.new('L', (height * big * 6, height * big * 5), 0)
        ImageDraw.Draw(r).text((height * big, height * big), ref, font=f, fill=255)
        rb = r.getbbox()
        s = height / (rb[3] - rb[1])
    w2 = max(1, round(im.size[0] * s))
    if w2 > max_w:
        w2 = max_w
    return im.resize((w2, max(1, round(im.size[1] * s))), Image.LANCZOS)


def label(suffix):
    text, path, var, shear, small = LABELS[suffix]
    canvas = Image.new('L', (160, 128), 0)
    if small:
        main = text_mask(text, path, var, 33, shear)
        canvas.paste(main, ((160 - main.size[0]) // 2, 42), main)
        sub = text_mask('CATCH', *LATIN, 11, 0, tracking=1.5)
        canvas.paste(sub, ((160 - sub.size[0]) // 2, 86), sub)
    else:
        main = text_mask(text, path, var, 28, ref='H')
        top = 53 if text.isupper() else 53  # (the capitals' top; descenders go below)
        canvas.paste(main, ((160 - main.size[0]) // 2, top), main)
    t01 = canvas
    shadow = t01.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2.2))
    t00 = Image.eval(shadow, lambda v: 255 - v)
    return t00, t01


def pokeball_icon(size, cx, cy, r, lines=True):
    """The icon's silhouette: a Poké Ball (band and button cut out) with three motion lines on its left: a throw."""
    big = 4
    im = Image.new('L', (size * big, size * big), 0)
    d = ImageDraw.Draw(im)
    cx, cy, r = cx * big, cy * big, r * big
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    band = r * 0.16
    d.rectangle((cx - r - 2, cy - band, cx + r + 2, cy + band), fill=0)
    br = r * 0.36
    d.ellipse((cx - br, cy - br, cx + br, cy + br), fill=0)
    br2 = r * 0.22
    d.ellipse((cx - br2, cy - br2, cx + br2, cy + br2), fill=255)
    if lines:
        lw = r * 0.16
        for k, (dy, ln) in enumerate(((-0.55, 0.55), (0.0, 0.8), (0.55, 0.55))):
            y = cy + dy * r
            x1 = cx - r * 1.18
            x0 = x1 - ln * r
            d.rounded_rectangle((x0, y - lw / 2, x1, y + lw / 2), radius=lw / 2, fill=255)
    im = im.rotate(-18, resample=Image.BICUBIC, center=(cx, cy))
    return im.resize((size, size), Image.LANCZOS)


def circle():
    """The button's face, 100x100: a pale red disc with a thin red ring and a white rim, and the icon in a red gradient
    (top half deeper), as the game's FIGHT / BAG / RUN faces."""
    n = 100
    big = 4
    yy, xx = np.mgrid[0:n * big, 0:n * big] / big
    c = (n - 1) / 2
    dist = np.hypot(xx - c, yy - c)
    disc = np.clip(49.2 - dist, 0, 1)
    # background: pale, lighter at the top left
    t = np.clip((xx + yy) / (2 * n), 0, 1)[..., None]
    bg = (1 - t) * np.array([255, 246, 246]) + t * np.array([255, 214, 218])
    # soft white clouds at the bottom left, as the game's watercolour
    rng = np.random.default_rng(7)
    cloud = np.zeros_like(dist)
    for _ in range(14):
        px, py, pr = rng.uniform(5, 45), rng.uniform(55, 95), rng.uniform(5, 13)
        cloud = np.maximum(cloud, np.clip(1 - np.hypot(xx - px, yy - py) / pr, 0, 1))
    bg = bg * (1 - 0.55 * cloud[..., None]) + 255 * 0.55 * cloud[..., None]
    ring = np.clip(1 - abs(dist - 45.2) / 1.1, 0, 1)[..., None]
    rim = np.clip(1 - abs(dist - 48.4) / 1.0, 0, 1)[..., None]
    rgb = bg * (1 - ring) + RED * ring
    rgb = rgb * (1 - rim) + 255 * rim
    # the icon, cropped by the ring like the game's
    icon = np.asarray(pokeball_icon(n * big, 56 * big, 56 * big, 30 * big)) / 255.0
    icon = icon * np.clip(43.5 - dist, 0, 1)
    g = np.clip((yy - 20) / 70, 0, 1)[..., None]
    icol = (1 - g) * DEEP + g * RED
    rgb = rgb * (1 - icon[..., None]) + icol * icon[..., None]
    a = disc * 255
    img = np.dstack([rgb, a]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, 'RGBA').resize((n, n), Image.LANCZOS)


def icon_mask():
    """The glow layer's mask, 200x200 white on black, centred as the game's."""
    return pokeball_icon(200, 104, 100, 52)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    circle().save(os.path.join(out, 'off.png'))
    icon_mask().save(os.path.join(out, 'icon.png'))
    sheet = Image.new('RGB', (160 * 2, 128 * len(LABELS) + 200), (60, 60, 60))
    sheet.paste(circle().resize((200, 200), Image.LANCZOS), (0, 0), circle().resize((200, 200), Image.LANCZOS))
    sheet.paste(icon_mask().resize((100, 100)), (210, 50))
    for i, sfx in enumerate(LABELS):
        t00, t01 = label(sfx)
        t00.save(os.path.join(out, f'text_00{sfx or "_jpn"}.png'))
        t01.save(os.path.join(out, f'text_01{sfx or "_jpn"}.png'))
        sheet.paste(t00.convert('RGB'), (0, 200 + 128 * i))
        sheet.paste(t01.convert('RGB'), (160, 200 + 128 * i))
    sheet.save(os.path.join(out, 'contact.png'))
    if '--header' in sys.argv:
        header(out, sys.argv[sys.argv.index('--header') + 1])


def header(out, path):
    """The PNGs as C arrays (optimised: palette-free 8-bit grey / RGBA, zlib level 9)."""
    names = ['off', 'icon'] + [f'text_0{k}{sfx or "_jpn"}' for sfx in LABELS for k in (0, 1)]
    lines = ["// Made by tools/catch_button_art.py --header: the CATCH button's artwork (PNG), drawn by that script.",
             '#pragma once', '#include <stddef.h>', '#include <stdint.h>', '']
    table = []
    for n in names:
        im = Image.open(os.path.join(out, n + '.png'))
        import io
        b = io.BytesIO()
        im.save(b, 'PNG', optimize=True)
        data = b.getvalue()
        ident = 'k_art_' + n
        lines.append(f'static const uint8_t {ident}[{len(data)}] = {{')
        for i in range(0, len(data), 24):
            lines.append('    ' + ', '.join('0x%02x' % x for x in data[i:i + 24]) + ',')
        lines.append('};')
        table.append(f'    {{"{n}", {ident}, sizeof {ident}}},')
    lines += ['', 'static const struct { const char *name; const uint8_t *png; size_t size; } k_catch_art[] = {'] + table + ['};', '']
    with open(path, 'w', newline=chr(10)) as f:
        f.write(chr(10).join(lines))


if __name__ == '__main__':
    main()
