#!/usr/bin/env bash
# walk_speed.sh <name> <fps> [ini lines...]: the player's walking speed in game time, from the camera (which follows
# the player): the walk down Viridian's road starts 5 s of game after the world does (present 1350 + 5 fps; the stick
# is ignored while the area's name shows), and LARECOMP_CAM_LOG gives the camera's position every frame. Prints how far
# it went (cm, on the ground plane) 0.5, 1 and 2 s after the walk began, and the speed between 0.5 and 1.5 s - the same
# at 30, 60 and 120 fps when the frame rate doesn't change the game's speed. Any LARECOMP_* passes through (e.g.
# LARECOMP_STEP_SCALE). A turbo run of tools/scenario_shot.sh (TAA on: the camera is read for its motion vectors).
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
name=$1; fps=$2; shift 2
w=$((1350 + 5 * fps))
cd "$R"
LARECOMP_CAM_LOG=1 INPUT_EXTRA=",$w:STICK_DOWN:3000" SHOTS="$w" LAST=$((w + 2 * fps + 1)) \
    timeout 300 bash tools/scenario_shot.sh "$name" "fps = $fps" "aa = taa" "$@" > /dev/null 2>&1
./.venv/Scripts/python - "build/scenario/$name.log" "$w" "$fps" "$name" <<'EOF'
import sys, re, math
log, w, fps, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pos = {}
for l in open(log, errors='replace'):
    m = re.match(r'\[cam\] present (\d+) pos (\S+) (\S+) (\S+)', l)
    if m: pos[int(m.group(1))] = tuple(map(float, m.group(2, 3, 4)))
def at(t):  # the position t s after the walk began (the nearest frame logged)
    n = w + round(t * fps)
    for d in range(4):
        for k in (n - d, n + d):
            if k in pos: return pos[k]
    return None
p0 = at(0)
if not p0: print(name, 'no camera log'); sys.exit(1)
dist = lambda p: math.hypot(p[0] - p0[0], p[2] - p0[2]) if p else float('nan')
a, b = at(0.5), at(1.5)
speed = math.hypot(b[0] - a[0], b[2] - a[2]) if a and b else float('nan')
print(f'{name}: {dist(at(0.5)):.0f} / {dist(at(1)):.0f} / {dist(at(2)):.0f} cm after 0.5 / 1 / 2 s; {speed:.0f} cm/s from 0.5 to 1.5 s')
EOF
