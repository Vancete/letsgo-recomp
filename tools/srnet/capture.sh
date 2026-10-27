#!/usr/bin/env bash
# capture.sh <data folder> [run...]: the tiny network's training sequences, from the installed game (scenario_shot.sh's
# scripted runs, Pallet Town) rendered at 4 x a 1920 x 1080 window, each real frame from 1250 to 1850 written by the
# runtime (LARECOMP_SRCAP: exec.c srcap_run) into <data folder>/<run>/. About 17 MB a frame, 10 GB a run.
# Runs: walk (left, rest, up, right), down (down, rest, right), menu (the menu opened and closed between walks),
# diag (diagonals), zigzag (short turns), val (mixed: kept for validation).
set -u
DATA=${1:?data folder}; shift
RUNS=${*:-walk down menu diag zigzag val}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
declare -A IN=(
    [walk]=",1250:STICK_LEFT:200,1500:STICK_UP:200,1760:STICK_RIGHT:120"
    [down]=",1250:STICK_DOWN:220,1520:STICK_RIGHT:260"
    [menu]=",1250:STICK_LEFT:80,1360:X:6,1450:B:6,1530:STICK_UP:120,1680:X:6,1760:B:6"
    [diag]=",1250:STICK_45:160,1420:STICK_225:160,1600:STICK_135:200"
    [zigzag]=",1250:STICK_LEFT:40,1290:STICK_UP:40,1330:STICK_LEFT:40,1370:STICK_UP:40,1440:STICK_RIGHT:60,1500:STICK_DOWN:60,1600:STICK_315:120,1760:STICK_90:60"
    [val]=",1250:STICK_UP:100,1380:STICK_RIGHT:100,1500:X:6,1580:B:6,1650:STICK_270:150"
)
for r in $RUNS; do
    mkdir -p "$DATA/$r"
    rm -f "$DATA/$r"/f*.bin
    (cd "$HERE" && GAME=${GAME:-/d/LetsGoRecomp/LetsGoRecomp} OUT="$DATA/_logs" LARECOMP_WINDOW=1920x1080 LARECOMP_SKIP_CONTROLLER=1 \
        LARECOMP_SCALE=4 LARECOMP_SRCAP="$DATA/$r" LARECOMP_SRCAP_FROM=1250 LARECOMP_SRCAP_TO=1850 \
        INPUT_EXTRA="${IN[$r]}" SHOTS=1850 LAST=1860 tools/scenario_shot.sh "cap_$r" "upscaler = dittosr_exp2" > /dev/null)
    echo "$r: $(ls "$DATA/$r" | wc -l) frames"
done
