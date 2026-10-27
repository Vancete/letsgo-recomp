#!/usr/bin/env bash
# measure.sh <out> [name:upscaler ...]: the upscalers compared at 4K (scenario_shot.sh's runs, the installed game), at
# Performance (2x) and Ultra Performance (3x). Per upscaler and mode: t<mode>_<name>/ - at rest, GPU profiles of 6
# frames and 6 consecutive window images (shimmer) plus the window at 1612 (still detail); m<mode>_<name>/ - moving
# (fine noise in flat areas); r<mode>_<name>/ - the game's menu opened and closed (settling). nat/: the reference,
# DittoSR at Native AA. Then: python metrics.py <out>.
set -u
O=${1:?out}; shift
VARIANTS=${*:-gsr2:gsr2 x2:dittosr_exp2 net:x2net dexp:dittosr_exp dit:dittosr}
R=$(cd "$(dirname "$0")/../.." && pwd)
D=${GAME:-/d/LetsGoRecomp/LetsGoRecomp}
mkdir -p "$O"
cd "$R" || exit 1
run() {  # run <name> <want images> <preset> <upscaler> <env...>: retried when the captures came out short (a locked file)
    local n=$1 want=$2 pre=$3 up=$4; shift 4
    for i in 1 2 3; do
        env "$@" LARECOMP_WINDOW=3840x2160 GAME=$D OUT=$O LARECOMP_SKIP_CONTROLLER=1 SHOTS=1610 LAST=1612 timeout 600 \
            bash tools/scenario_shot.sh "$n" "aa = smaa" "upscaler = $up" "fsr3_preset = $pre" > "$O/$n.txt" 2>&1
        [ "$(ls "$O/$n" 2>/dev/null | wc -l)" -ge "$want" ] && return
    done
    echo "$n: short ($(ls "$O/$n" 2>/dev/null | wc -l) images)"
}
[ -d "$O/nat" ] || run nat 2 native dittosr
for pre in performance ultra; do
    for v in $VARIANTS; do
        IFS=: read -r n up <<< "$v"
        run "t${pre}_$n" 8 $pre $up LARECOMP_GPUPROF_AT=1600,1601,1602,1603,1604,1605 LARECOMP_WINDOW_SHOTS=1596,1597,1598,1599,1600,1601
        run "m${pre}_$n" 2 $pre $up INPUT_EXTRA=",1500:STICK_LEFT:200"
        run "r${pre}_$n" 22 $pre $up INPUT_EXTRA=",1500:X:6,1560:B:6" LARECOMP_WINDOW_SHOTS=$(seq -s, 1560 1580),1600
        echo "$pre $n done"
    done
done
