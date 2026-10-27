#!/usr/bin/env bash
# measure.sh <out> [name:upscaler[:ENV=v,ENV=v...] ...]: the upscalers compared at 4K (scenario_shot.sh's runs, the
# installed game), at the modes in PRESETS (default "performance ultra"; quality, balanced too). Per upscaler and mode:
# p<mode>_<name> - GPU profiles of 6 frames at rest; t<mode>_<name>/ - 6 consecutive window images at rest (shimmer)
# plus the window at 1612 (still detail); m<mode>_<name>/ - moving (fine noise in flat areas); r<mode>_<name>/ - the
# game's menu opened and closed (settling; SETTLE=0 leaves it out). The references: nat/ and pnat - native 4K, no
# upscaler (SMAA); natfsr4/ - DittoSR (FSR4) at Native AA (NATFSR4=0 leaves it out). WIN: the window's size (default
# 3840x2160; metrics.py and sheet.py scale their crops to it), or "full" - full screen at the display's resolution.
# Then: python metrics.py <out>.
set -u
O=${1:?out}; shift
VARIANTS=${*:-fsr3:fsr3 ditto:dittosr fsr4:dittosr_fsr4 net:x2net}
R=$(cd "$(dirname "$0")/../.." && pwd)
D=${GAME:-/d/LetsGoRecomp/LetsGoRecomp}
WIN=${WIN:-3840x2160}
if [ "$WIN" = full ]; then WENV=(); WINI="fullscreen = 1"; else WENV=(LARECOMP_WINDOW=$WIN); WINI="fullscreen = 0"; fi
mkdir -p "$O"
cd "$R" || exit 1
run() {  # run <name> <want images> <preset> <upscaler> <env...>: retried when the captures came out short (a locked file)
    local n=$1 want=$2 pre=$3 up=$4; shift 4
    for i in 1 2 3; do
        env "$@" "${WENV[@]}" GAME=$D OUT=$O LARECOMP_SKIP_CONTROLLER=1 SHOTS=1610 LAST=1612 timeout 600 \
            bash tools/scenario_shot.sh "$n" "aa = smaa" "upscaler = $up" "fsr3_preset = $pre" "$WINI" > "$O/$n.txt" 2>&1
        [ "$(ls "$O/$n" 2>/dev/null | wc -l)" -ge "$want" ] && return
    done
    echo "$n: short ($(ls "$O/$n" 2>/dev/null | wc -l) images)"
}
[ -d "$O/nat" ] || run nat 8 native bilinear LARECOMP_WINDOW_SHOTS=1596,1597,1598,1599,1600,1601
[ -f "$O/pnat.log" ] || run pnat 2 native bilinear LARECOMP_GPUPROF_AT=1600,1601,1602,1603,1604,1605
[ -d "$O/mnat" ] || run mnat 2 native bilinear INPUT_EXTRA=",1500:STICK_LEFT:200"
[ -d "$O/natfsr4" ] || [ "${NATFSR4:-1}" = 0 ] || run natfsr4 2 native dittosr_fsr4
for pre in ${PRESETS:-performance ultra}; do
    for v in $VARIANTS; do
        IFS=: read -r n up ev <<< "$v"
        E=(${ev//,/ })
        # (the profile in a run of its own: the window images' readbacks idle the GPU and its clock drops)
        run "p${pre}_$n" 2 $pre $up "${E[@]}" LARECOMP_GPUPROF_AT=1600,1601,1602,1603,1604,1605
        run "t${pre}_$n" 8 $pre $up "${E[@]}" LARECOMP_WINDOW_SHOTS=1596,1597,1598,1599,1600,1601
        run "m${pre}_$n" 2 $pre $up "${E[@]}" INPUT_EXTRA=",1500:STICK_LEFT:200"
        [ "${SETTLE:-1}" = 0 ] || run "r${pre}_$n" 22 $pre $up "${E[@]}" INPUT_EXTRA=",1500:X:6,1560:B:6" LARECOMP_WINDOW_SHOTS=$(seq -s, 1560 1580),1600
        echo "$pre $n done"
    done
done
