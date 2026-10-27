"""The GPU time of DittoSR's passes per run (LARECOMP_GPUPROF_AT logs): python times.py <log>...

Per log, the median over the profiled frames (the first left out) of the rows "ours: DittoSR" (the convert pass and
x2_change.frag), "ours: DittoSR upscale", "ours: DittoSR lines, interface" (the final pass), their sum, and the
scene's main pass (the same work in every run: a gauge of the GPU's clock)."""
import re
import statistics
import sys

ROWS = ["ours: DittoSR", "ours: DittoSR upscale", "ours: DittoSR lines, interface"]


def parse(log):
    frames, cur = [], None
    for line in open(log, encoding="utf-8", errors="replace"):
        if "[gpuprof]" not in line:
            continue
        if "marks," in line:
            cur = {"scene": 0.0}
            frames.append(cur)
            continue
        m = re.match(r".*\[gpuprof\]\s+([\d.]+) ms\s+[\d.]+%\s+(.*)", line)
        if not m or cur is None:
            continue
        t, what = float(m.group(1)), m.group(2).strip()
        if what in ROWS:
            cur[what] = cur.get(what, 0.0) + t
        dm = re.match(r"pass: 1 color \d+x\d+ .* (\d+) draws", what)
        if dm and int(dm.group(1)) > 100:
            cur["scene"] = max(cur["scene"], t)
    return [f for f in frames[1:] if all(r in f for r in ROWS)]


def main():
    print(f"{'run':24s} {'convert':>8s} {'upscale':>8s} {'final':>8s} {'total':>8s} {'scene':>8s}  (ms, median)")
    for log in sys.argv[1:]:
        fr = parse(log)
        if not fr:
            print(f"{log}: no frames")
            continue
        med = {k: statistics.median(f[k] for f in fr) for k in ROWS + ["scene"]}
        tot = statistics.median(sum(f[k] for k in ROWS) for f in fr)
        name = log.replace("\\", "/").split("/")[-1][:-4]
        print(f"{name:24s} {med[ROWS[0]]:8.3f} {med[ROWS[1]]:8.3f} {med[ROWS[2]]:8.3f} {tot:8.3f} {med['scene']:8.3f}  ({len(fr)} frames)")


if __name__ == "__main__":
    main()
