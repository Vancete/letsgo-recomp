"""A contact sheet of measure.sh's runs: python sheet.py <out> <mode: performance|ultra> <sheet.png>

One row per upscaler (and Native AA), three crops at 2x zoom (nearest): the player at rest, the fence at rest, the
player moving (the camera follows the player: the same place on screen in every run)."""
import os
import sys

from PIL import Image, ImageDraw

CROPS = [("rest: player", "t", (1780, 860, 320, 320)), ("rest: fence", "t", (1350, 250, 320, 320)),
         ("moving: player", "m", (1760, 900, 320, 320))]
ROWS = [("gsr2", "Snapdragon GSR 2"), ("x2", "Experiment 2"), ("net", "Experiment 2 Net"), ("dexp", "DittoSR Experiment"),
        ("dit", "DittoSR"), ("nat", "Native AA (DittoSR)")]


def main():
    out, pre, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    z, lab = 2, 200
    sheet = Image.new("RGB", (lab + len(CROPS) * 320 * z, len(ROWS) * 320 * z + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for j, (title, _, _) in enumerate(CROPS):
        d.text((lab + j * 320 * z + 8, 8), title, fill=(230, 230, 230))
    for i, (n, name) in enumerate(ROWS):
        d.text((8, 30 + i * 320 * z + 8), name, fill=(230, 230, 230))
        for j, (_, kind, (x, y, w, h)) in enumerate(CROPS):
            run = "nat" if n == "nat" else f"{kind}{pre}_{n}"
            if n == "nat" and kind == "m":
                continue
            p = os.path.join(out, run, "window.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB").crop((x, y, x + w, y + h)).resize((w * z, h * z), Image.NEAREST)
            sheet.paste(im, (lab + j * 320 * z, 30 + i * 320 * z))
    sheet.save(dst)


if __name__ == "__main__":
    main()
