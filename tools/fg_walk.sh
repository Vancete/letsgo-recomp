#!/usr/bin/env bash
# fg_walk.sh <name> <frames, e.g. "1480 1520"> [ini lines]: the scripted run of tools/scenario_shot.sh at real speed (fps
# 30, not turbo), walking (WALK="<frame>:STICK_DOWN:<frames>", default up from 1500); the window as it reached the
# screen captured from the desktop (tools/winshot.ps1) at those frames -> $OUT/<name>/win_N.png. With
# LARECOMP_FG_ONLY_GENERATED=1: frame generation's frames alone. Opens the game window; do not cover it.
set -u
SP=$(cd "$(dirname "$0")" && pwd)
GAME=${GAME:-/d/LetsGoRecomp/LetsGoRecomp}
OUT=${OUT:-$(pwd)/build/scenario}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
name=$1; shots=$2; shift 2
mkdir -p "$OUT/$name"; rm -f "$OUT/$name"/*.png
cd "$GAME" || exit 1
rm -rf "$OUT/save_$name"; mkdir -p "$OUT/save_$name"; cp -r save-pikachu/* "$OUT/save_$name/"
ini="$OUT/$name.ini"
printf 'fullscreen = 0\nfps = 30\nrender_height = 0\nsmaa = 0\naa = off\nresolution = match\nrender_thread = 1\n' > "$ini"
for kv in "$@"; do echo "$kv" >> "$ini"; done
LARECOMP_FPS_INTERVAL=0.25 LARECOMP_SETTINGS="$ini" LARECOMP_INPUT="400:A:6,500:A:6,1150:A:6,1300:A:6,${WALK:-1500:STICK_UP:600}" \
  LARECOMP_NO_USER_INPUT=1 LARECOMP_SAVE_DIR="$OUT/save_$name" LARECOMP_NO_AUDIO=1 \
  ./LetsGoRecomp.exe extracted --play > "$OUT/$name.log" 2>&1 &
sleep 3
pid=$(powershell -NoProfile -c "(Get-Process LetsGoRecomp | Sort-Object StartTime -Descending | Select-Object -First 1).Id")
for f in $shots; do
  for i in $(seq 1 600); do
    cur=$(grep -oE "frame [0-9]+ [0-9.]+ fps" "$OUT/$name.log" | tail -1 | awk '{print $2}')
    [ -n "$cur" ] && [ "$cur" -ge "$f" ] && break
    sleep 0.1
  done
  powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$SP/winshot.ps1")" -ProcId "$pid" -Out "$(cygpath -w "$OUT/$name/win_$f.png")"
done
powershell -NoProfile -c "Stop-Process -Id $pid -Force"
grep -oE "window [0-9.]+ fps" "$OUT/$name.log" | tail -3
