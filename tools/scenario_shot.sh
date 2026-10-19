#!/usr/bin/env bash
# scenario_shot.sh <name> [ini lines...]: a scripted run of an installed game (release layout: LetsGoRecomp.exe,
# extracted/, save-pikachu/) to Pallet Town in turbo, with a copy of the save (the player's own is never touched) and a
# settings file of its own (the player's larecomp.ini is never touched). Frame ~1150 on: gameplay.
#
# env: GAME=<the game's folder> (default /d/LetsGoRecomp/LetsGoRecomp), OUT=<results> (default build/scenario),
#      INPUT_EXTRA=",<frame>:<BUTTON>:<frames>,..." more input (STICK_RIGHT, STICK_DOWN, A, X...),
#      SHOTS=<frame>,... game images (default 1600), LAST=<frame> window image and exit (default 1610),
#      any other LARECOMP_* (LARECOMP_MV_DUMP, LARECOMP_SSAO_DUMP, LARECOMP_GPUPROF_AT...) passes through.
# ini lines: e.g. "ssao_level = 3" "gi_level = 1" "aa = taa" "aa = fsr3" "fsr3_preset = performance" "frame_gen = 2"
# (later lines win; "upscaler = fsr3" of older runs now means "aa = fsr3").
# Results: $OUT/<name>/ (frame_N.png, window.png, dumps) and $OUT/<name>.log.
# Note: run it on its own (not chained after make in the same command: the environment was lost that way), and it
# opens a game window on the desktop for some seconds - do not close it.
set -u
GAME=${GAME:-/d/LetsGoRecomp/LetsGoRecomp}
OUT=${OUT:-$(pwd)/build/scenario}
name=$1; shift
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$GAME" || exit 1
rm -rf "$OUT/save" build "$OUT/$name"
mkdir -p "$OUT/save" build/frames "$OUT/$name"
cp -r save-pikachu/* "$OUT/save/"
cat > "$OUT/$name.ini" <<'EOF'
fullscreen = 0
fps = 30
render_height = 0
smaa = 0
aa = off
upscaler = bilinear
resolution = match
render_thread = 1
EOF
for kv in "$@"; do echo "$kv" >> "$OUT/$name.ini"; done
LARECOMP_FPS=0 LARECOMP_SETTINGS="$OUT/$name.ini" LARECOMP_INPUT="400:A:6,500:A:6,1150:A:6,1300:A:6${INPUT_EXTRA:-}" \
    LARECOMP_NO_USER_INPUT=1 LARECOMP_SAVE_DIR="$OUT/save" LARECOMP_NO_AUDIO=1 LARECOMP_SHOT_AT="${SHOTS:-1600}" \
    LARECOMP_WINDOW_SHOT="${LAST:-1610}" LARECOMP_EXIT_AFTER_SHOT=1 timeout 300 ./LetsGoRecomp.exe extracted --play > "$OUT/$name.log" 2>&1
echo "$name rc=$?"
grep -E "\[ssao\]|\[gi\]|\[taa\]|\[mv\]|\[fsr3\]|\[post\]|FATAL|exception|window closed" "$OUT/$name.log" | head -12
cp build/frames/*.png "$OUT/$name/" 2>/dev/null
cp build/window.png "$OUT/$name/window.png" 2>/dev/null
echo "$name: $(ls "$OUT/$name" | wc -l) images in $OUT/$name"
rm -rf build
