"""A contact sheet of measure.sh's runs: python sheet.py <out> <mode: quality|performance|ultra> <sheet.png>

One row per upscaler (native 4K first), three crops at 2x zoom (nearest): the player at rest, the fence at rest, the
player moving (the camera follows the player: the same place on screen in every run)."""
import os
import sys

from PIL import Image, ImageDraw

CROPS = [("rest: player", "t", (1780, 860, 320, 320)), ("rest: fence", "t", (1350, 250, 320, 320)),
         ("moving: player", "m", (1760, 900, 320, 320))]
ROWS = [("nat", "Native (SMAA)"), ("fsr3", "FSR 3.1"), ("ditto", "DittoSR"), ("fsr4", "DittoSR (FSR4)"),
        ("net", "DittoSR Net"), ("stud", "DittoSR Net (student)"), ("stud1", "Net (student, scale 1)"), ("stud2m", "Net v2 (mixed teacher)"), ("stud2", "Net v2 (student)")]


def main():
    out, pre, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    z, lab = 2, 200
    rows = [r for r in ROWS if r[0] == "nat" or os.path.isdir(os.path.join(out, f"t{pre}_{r[0]}"))]  # (those measured)
    sheet = Image.new("RGB", (lab + len(CROPS) * 320 * z, len(rows) * 320 * z + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for j, (title, _, _) in enumerate(CROPS):
        d.text((lab + j * 320 * z + 8, 8), title, fill=(230, 230, 230))
    for i, (n, name) in enumerate(rows):
        d.text((8, 30 + i * 320 * z + 8), name, fill=(230, 230, 230))
        for j, (_, kind, (x, y, w, h)) in enumerate(CROPS):
            run = ("mnat" if kind == "m" else "nat") if n == "nat" else f"{kind}{pre}_{n}"
            p = os.path.join(out, run, "window.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            k = im.width / 3840  # (the crops at 4K: the same place at other sizes, the zoom up to the same size)
            im = im.crop(tuple(int(round(v * k)) for v in (x, y, x + w, y + h))).resize((w * z, h * z), Image.NEAREST)
            sheet.paste(im, (lab + j * 320 * z, 30 + i * 320 * z))
    sheet.save(dst)


if __name__ == "__main__":
    main()
