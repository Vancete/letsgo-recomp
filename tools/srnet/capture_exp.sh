#!/usr/bin/env bash
# capture_exp.sh <data folder> [run...]: the teacher's pairs (LOCAL research only - the captures and anything trained on
# them never leave this machine): the installed game played with upscaler = dittosr_exp (DittoSR Experiment, at
# LARECOMP_DITTO_EXP_SCALE, default here 0.75) in a 1920 x 1080 window at PRESET (default ultra), each real frame from
# 1250 to 1850 written by the runtime (LARECOMP_EXPCAP: exec.c expcap_run) into <data folder>/<run>/ - capture.sh's
# scripted walks. About 11 MB a frame at 3x.
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
        LARECOMP_DITTO_EXP_SCALE=${LARECOMP_DITTO_EXP_SCALE:-0.75} LARECOMP_EXPCAP="$DATA/$r" LARECOMP_SRCAP_FROM=1250 \
        LARECOMP_SRCAP_TO=1850 INPUT_EXTRA="${IN[$r]}" SHOTS=1850 LAST=1860 tools/scenario_shot.sh "cap_$r" \
        "upscaler = dittosr_exp" "fsr3_preset = ${PRESET:-ultra}" > /dev/null)
    echo "$r: $(ls "$DATA/$r" | wc -l) frames"
done
