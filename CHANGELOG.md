# Changelog

## Unreleased

- **Fixes walking speed at 60 and 120 fps**: the player walked at ~85 % of the speed at 60 fps and ~53 % at 120. The
  player moves by the walking animation's root motion (time-based) plus, once per world update, an offset of its own
  (f_700021b6f0 of Pikachu 1.0.0: -2 cm while running, sized for a 33.3 ms update) - 60 cm/s off the animation's
  450.8 at 30 fps, 120 at 60, 240 at 120. The same function turns the player toward the stick with a slerp of t = 0.5
  per update, so turning was 2x / 4x as fast. Now, in the world at 60 / 120 fps, that offset is scaled by the update's
  share of 33.3 ms and the slerp's t becomes 1 - (1 - t)^share. Measured in game time from the camera
  (`tools/walk_speed.sh`): 391 cm/s at 30, 60 and 120 fps (before: 391 / 331 / 211). The functions are found by their
  code near Pikachu 1.0.0's address (Eevee's should be a few instructions away - not tested). Found with new research
  aids: `LARECOMP_CAM_LOG`, `LARECOMP_STEP_SCALE`, `LARECOMP_FARG_SCALE`, `LARECOMP_FIND_MOVER`, memwatch's
  `LARECOMP_WATCH_LIKE` / `_AT` / `_MAX` (docs/DEPURACAO.md, docs/PESQUISA-60FPS.md).
- **Frame rate limit in the F12 menu**: 30 fps (original) / 60 fps / 120 fps (experimental), switched while playing.
  120 is the 60 fps mode with the world's step at 8.33 ms; menus stay at 30 with their in-between image (60), battles
  and captures at 30. The mode's hooks are now added when it is first turned on (on the game thread; a hook's function
  is written before its address, for threads looking it up meanwhile). The start screen's 60 fps box keeps 120 when on.
  `larecomp.ini`: `fps = 120`.
  - The player walked slower the higher the rate (85 % / 53 %): fixed, see "Fixes walking speed" above.
- **Fixes the extreme ghosting of frame generation on vertical motion**: the present turns the game image upside down
  (its origin is the lower left), and frame generation works on the window image - but it was given the depth and the
  motion vectors the right way up. Sideways the two agree; up and down every vector pointed the wrong way, and the
  generated frame doubled the whole scene vertically while walking up or down. Frame generation now gets them flipped
  (the vectors' y and the jitter's negated). Checked on the generated frames alone, captured from the desktop while
  walking down through Pallet Town (`tools/fg_walk.sh`). The F12 menu's frame generation is Off / 2x only now.
- The post passes' pipeline cache had 16 entries for about 30 passes: with FSR 3.1, frame generation, SSAO and GI on,
  some pipelines were made again every frame. 64 now.
- **F12 menu reorganised**:
  - **Anti-aliasing**: Off / SMAA / TAA / FSR 3.1. FSR 3.1 is the last choice instead of an upscaler of its own (with
    its mode below it); with any other choice a lower render resolution is scaled bilinearly (FSR 1 is gone from the
    menu; it stays as the fallback when FSR 3.1's DLL is missing). `larecomp.ini`: `aa = fsr3`; an older
    `upscaler = fsr3` reads as it.
  - **Frame generation**: Off / 2x, with FSR 3.1 anti-aliasing, switched live (AMD's swapchain is made or dropped at the
    next frame; its queues are now created with the device whenever the DLL is there). FSR 3.1 generates one frame
    between two (several is FSR 4's, DirectX 12 only). `frame_gen = 2`.
  - **SSAO and GI**: quality levels Off (default) / Low / Medium / High instead of a box, and their parameters are gone
    from the advanced page (its debug views stay, under Debug). High is what they had before. SSAO: 16 / 12 / 8 samples
    (the shader takes the count now). GI: 4 rays x 8 steps at half the scene's size / 3 x 6 at half / 2 x 6 at a third.
    The strength, reach and accumulation are the same at every level. 720p, RTX 5070 Ti, both on: +0.21 / +0.19 / +0.17
    ms. `ssao_level`, `gi_level` (0-3); an older `ssao = 1` / `gi = 1` reads as high.
- **FSR 3.1 frame generation** (`frame_gen = 1` in `larecomp.ini`, off by default; with the FSR 3.1 upscaler; read when
  the game starts): AMD FidelityFX generates a frame between each two of the game's, on the swapchain below. At
  `fps = 60` the window gets 120 images a second while playing, and 60 in menus and cutscenes (where the game runs at
  30); the game's own in-between image of the 60 fps mode is off then. Interface excluded: the window image without the
  interface (FSR 3's output where the interface mask is clear, the scene from before the interface where it is not) is
  handed over each frame as FidelityFX's "HUD-less" image, so the interface is taken from the real frame and not warped.
  The depth, the motion vectors and the jitter are FSR 3.1's. Where FSR 3.1 does not run (2D screens without the 3D
  scene), no frame is generated. AMD's presenter thread keeps only the latest frame; two dropped in a row stalled the
  GPU for good (1 run in 3 of the scripted turbo test, after a slow frame), so a present now waits until the presenter
  has made one since the frame before (at most 100 ms; 18 turbo runs without a stall since).
- **Fixes FSR 3.1's interface mask while playing**: the mask (the pixels the interface draws, which FSR 3.1 treats
  apart) was taken only from draws into the presented image, but while playing the game draws its interface at the
  render size into the image its last pass grades into the presented one - and the 2x interface images and the
  high-resolution text read as the scene's (scaled) images. Now the copy is made of that image, and only a scaled render
  target counts as the scene's.
- **AMD FidelityFX swapchain, the first step towards frame generation** (`ffx_swapchain = 1` in `larecomp.ini`, off by
  default; read when the game starts): the window's swapchain is FidelityFX's (SDK 1.1.4, the same signed
  `amd_fidelityfx_vk.dll` as FSR 3.1). The game draws into FidelityFX's images and FidelityFX copies them into the real
  swapchain on a present queue of its own. No frame is generated yet. The device gets three more queues for it
  (present, image acquire, async compute; RTX 5070 Ti: families 0, 1 and 2). Without a free queue or the DLL, the
  driver's swapchain is used as before. Tested in Pallet Town at 30 and 60 fps with FSR 3.1, TAA, SSAO and GI, and in
  full screen at 4K (the swapchain made again): same images, same frame pacing.

## 0.4.0 (2026-10-19)

- **Fixes the world drawn wrong after a catch** (0.3.2): black trees with yellow stripes, bushes with a sign's texture
  or a flat colour, and no dirt path until the area changed. After a catch the game loads shaders again at addresses
  that held other shaders, and the renderer handed back the old translations; shaders are now known by their content
  too. Checked with automatic catch loops: 7 catches on Eevee and 9 on Pikachu.
- **2x interface checked before it is packed**: waifu2x once (in seven runs) exited normally with garbage for every
  image, and the menus' icons came out as noise. Each 2x image, taken back down to 1x, must now match the original
  (good ones differ by up to 7 of 255, that garbage by 66 or more); with more than 10% off, no pack is written and the
  start screen says to try again.
- **Fixes many 2x interface images showing wrong** (partly or wholly transparent): waifu2x-ncnn-vulkan 20250915
  loses the alpha of an RGBA image outside its last tiles (a fully opaque 512x288 image came out 25/32 transparent).
  Images are now upscaled as RGB only - with transparency, their colour and their alpha (as a grey image) separately,
  put back together when packed. Opaque images now go into BC1 (the pack: 667 -> 548 MB). Make them again once.
- **Better background blur and depth of field** (`blur_quality = 1`, on by default): the menus' blurred backdrop and the
  title screen's depth of field come from one post-processing chain of the game's - the HDR scene taken down to 1/2,
  1/3 and 1/6 of its size with a single bilinear tap, blurred there with 5- and 9-tap separable kernels, mixed back by
  each pixel's blur size with bilinear reads (blocky, diamond-shaped steps at 1440p and 4K). Seven of its shaders, known
  by the hash of their code, now read their textures better: a 13-tap downsample (Jimenez), four rotated-grid taps
  per blur tap, and a cubic B-spline wherever a smaller level is magnified (the full-size scene is read as before). The
  game's own weights, focus and blend are untouched. The bloom shares the downsample and one blur shader. Cost at
  1440p, RTX 5070 Ti: +0.27 ms while a menu or the depth of field is on screen (the composite 0.05 -> 0.22 ms), a few
  hundredths of a millisecond for the bloom otherwise.
- **Sharper interface (2x)**, optional: a start-screen button makes the game's menu and interface images again at
  twice their resolution from the player's own game files - nothing of the game ships with the app. Every texture of
  the interface archives (`bin/app/*.arc`, 3081 different images, 316 more in formats left out) is decoded, upscaled by
  waifu2x-ncnn-vulkan (CUnet, light denoising; downloaded once from GitHub, 35 MB, checked by its SHA-256) and
  compressed again (BC1 / BC3 / BC4 / BC5) into `cache/ui2x.pak` (~670 MB). In the game a texture is matched by the
  key of its content and drawn from its 2x version at the same place. RTX 5070 Ti: about a minute (14 s reading the
  images, 17 s upscaling, 25 s compressing). A box on the start screen turns them off (`ui2x = 0`). The Real-ESRGAN
  build tried first loses the device on RTX 50 GPUs (2022 ncnn); waifu2x's 2025 build runs there.
- **Fixes SSAO, GI, TAA and FSR 3.1 flickering in cutscenes and menus at 60 fps**: there the game runs at 30 and an
  in-between image is drawn by replaying the frame; that image shared the frame's number, so the scene effects (which
  run once per frame) skipped it and every other image on screen went out without occlusion, indirect light or
  anti-aliasing. The in-between image is now a frame of its own for them, in the order the images are rendered: its own
  jitter and history, its camera blended between the two frames as its draws are (motion vectors), and frame N's
  post-processing (FSR 3.1, SMAA, the colour pass) done before the replay overwrites its inputs and shown after it.
  Professor Oak's cutscene in Pallet Town, FSR 3.1 Native AA + SSAO + GI: the brightness jump between consecutive
  images 0.100 -> 0.007, the in-between and final images now as alike as with the effects off (0.2 vs 0.13).
- **Sharper text at high resolutions** (always on, nothing to set): the game rasterizes its text with the system font into
  a 2048x1024 glyph cache at sizes 22-36, which went blurry when stretched to 1440p or 4K. Every glyph the game caches is
  now rasterized again by the same engine (nn::fontll) at N x its size into an N x copy of the cache, which the renderer
  samples instead of the original (same coordinates, so placement and layout are the game's). N is one step per 720
  lines of the window when the game starts (720p: off, 1440p: 2x, 4K: 3x). Cost: one extra glyph rasterization per new glyph and only the changed part of
  the cache uploaded (one full upload when the cache is made). `runtime/hle/font_hires.c`.
- **In-game settings on F12**: volume, brightness, contrast and saturation, applied live and saved to `larecomp.ini`
  (`brightness`, `contrast`, `saturation`). From a controller: − and + together, then *Settings*.
  `LARECOMP_MENU_KEY=<key name>` picks another key (Steam takes F12 for screenshots). The colour pass runs on the
  whole image, the game's interface included, and only when a value is not 100%.
- **Smoother shadows** (F12, *Shadow filter*): the game samples its 2048² shadow map with nearest filtering, so each
  of its 11 taps is a hard in / out and shadow edges come out stair-stepped. *Bilinear* makes each tap a 2x2
  depth-compare filter; *Smooth* (the default) a 3x3 tent of four bilinear taps; *Game's* keeps the original. Same
  shadow-map resolution. Measured at 1280x720 on an RTX 5070 Ti: the scene pass from 0.16 to 0.62 ms (*Smooth*).
  `shadow_filter = game | bilinear | smooth` in `larecomp.ini`.
- **Camera motion vectors** (groundwork for temporal effects): the game's view and projection are read from its scene
  constant block each frame; between the scene's opaque and blended draws each pixel's depth is taken to the previous
  frame (previous view-projection x inverse of this one, in double) into an RG16F image. Static scenery only: the
  characters get the camera's motion, not their own. F12 *Show motion vectors (debug)*; checked with
  `tools/mv_check.py` (reprojecting the previous frame cuts the error by 80-90% while walking).
- **F12 menu**: brightness, contrast and saturation moved to *Advanced...* (an *Image* section); the first page keeps the
  volume, the effects' switches, the shadow filter, anti-aliasing and the upscaler. With **FSR 3.1** chosen, its mode
  sets the render resolution - AMD's presets: Native AA (1.0x, FSR as anti-aliasing), Quality (1.5x, the default),
  Balanced (1.7x), Performance (2.0x), Ultra Performance (3.0x) - instead of the start screen's resolution
  (`fsr3_preset = native | quality | balanced | performance | ultra`); TAA's sharpness slider also sets FSR 3.1's. A
  page taller than the screen (the advanced one at 720p) scrolls. The start screen's box is just "Anti-aliasing" (SMAA
  when on, unless TAA was chosen in F12).
- **Per-object motion vectors** (F12 *Advanced...*, on by default whenever motion vectors are used): the scene pass's
  depth-tested draws run a variant of the game's vertex shader that also computes the previous frame's position from
  that draw's previous constant buffers (the shader body emitted twice by the translator, `cb<k>` read as `pcb<k>`),
  and their fragment shaders write the velocity to an extra RG16F-class target. A draw is matched to the previous
  frame's by program, vertex buffer, draw parameters, game object and occurrence; a draw missing any previous block, or
  whose block jumps (another object paired, a cut), marks its velocity unknown and the camera's is used there.
  Measured walking (`tools/mv_check.py`, `MV_BOX` on the player): reprojection error on the player 22-24 -> 12
  (camera-only vectors moved the player like the background); whole image 1.87 -> 1.67.
- **TAA** (F12, *Anti-aliasing*: Off / SMAA / TAA; sharpness under *Advanced...*): the scene's draws on its depth are
  jittered (Halton 2, 3 in the vertex shader, after the position the velocity uses), the complete HDR scene is resolved
  just before the game's bloom (Catmull-Rom history, YCoCg variance clipping, the nearest velocity of the 3x3,
  depth-checked disocclusion, tone-mapped blending), RCAS at the present. The interface is not touched (SMAA blurred
  it). Frame-to-frame instability walking 10-17% lower than without; no trail behind the player.
- **FSR 3.1** (F12, *Upscaler*: Bilinear / FSR 1 / FSR 3.1), when the render resolution is below the window's: AMD
  FidelityFX SDK 1.1.4 through its C API and AMD's signed `amd_fidelityfx_vk.dll` (MIT; shipped next to the exe,
  loaded at run time - without it, FSR 1). The newest FSR with a Vulkan backend: the FSR SDK 2.x (FSR 4) is DirectX 12
  only. The scene is jittered with FSR's own sequence; FSR gets the game image, the scene depth, the motion vectors
  (camera + per object) and a mask of the interface (the pixels that changed after the frame's first interface draw).
  From 640x360 to 1280x720 it is visibly cleaner than FSR 1 (difference to native 5.14 vs 5.40). The backend asks for
  debug-utils and AMD marker functions this device lacks: they get no-ops; `...KHR` names fall back to Vulkan 1.1 core.
  Not yet checked: the interface mask over a moving scene (battles).
- **Indirect light (GI)**, experimental, off by default (F12 *Indirect light (GI)*; its parameters under *Advanced...*):
  one bounce of the visible light in screen space, after the design of FastGI (a Unity URP feature; the technique,
  rewritten for this renderer - no code taken from it). Between the scene's opaque and blended draws: normals rebuilt
  from the depth (the game has no normal buffer), rays traced at half / a third / a quarter of the resolution against
  the depth with the game's own projection (cosine-distributed, quadratic steps, the first crossing refined by
  bisection and validated: on the ray, facing it and the camera), the scene's colour where they hit (luminance
  clamped), two bilateral blurs, accumulation along the camera motion vectors (depth-validated, neighbourhood-clamped),
  an edge-aware upsample added onto the scene. Grass greens the curbs, fences and the foot of walls, the red roofs
  their eaves. Defaults: intensity 200%, radius 300, thickness 25 (the game's units are ~cm), 4 rays x 8 steps, half
  resolution. Measured at 1280x720 on an RTX 5070 Ti: +0.36 ms. Walking, the accumulated light has no trail behind the
  player (brighter / darker differences against single-frame GI symmetric) and much less grain.
- **F12 menu split**: the effects are switched on the first page (SSAO, GI, the shadow filter); their sliders and the
  debug views moved to *Advanced...*. SSAO defaults are now intensity 200%, radius 100.
- **SSAO accumulated over frames** (F12 *Accumulate over frames*, on by default with SSAO): the sample spiral turns
  every frame and the result is blended with the previous one fetched along the camera motion vectors (90% history),
  rejected where the stored distance is not the one the pixel had (uncovered ground, characters moving on their own)
  and clamped to the current 3x3 range. Checked walking and stopping against single-frame SSAO
  (`tools/ssao_temporal_check.py`): a first version with a 2% clamp margin left the player's contact shadow as a trail
  that never faded (2-4% of pixels darker); with the exact clamp the differences are symmetric noise (~1% darker, ~1%
  lighter) and nothing remains after stopping; high-frequency noise down 8-10%.
- The F12 debug views (the occlusion alone, the motion vectors) are shown at the present, in place of the game image:
  written into the scene they came out white or washed out by the game's fog, bloom and tone mapping.
- **Ambient occlusion (SSAO)**, experimental and off by default (F12: on / off, intensity, radius, a view of the
  occlusion alone): computed from the game's depth between its opaque and blended draws, with the game's own projection.

## 0.3.2 (2026-10-18)

- **Fixes the launcher forgetting your settings** (0.3.1): it saved `larecomp.ini` with some lines run together, so
  the next start went back to the defaults for 60 fps, the resolution, anti-aliasing, the translation, movie audio,
  the fps counter, the d-pad and wild battles. Fixed, and a file saved by 0.3.1 is read correctly: tested with one,
  every setting came back.
- **Game folders with accents or other non-Latin letters work** (for example a Windows user folder named with an
  accent). Tested with a folder named in Cyrillic and Portuguese letters: 0.3.1 could not open the game there, 0.3.2
  runs it.
- **Less memory reserved**: the game's threads no longer commit 256 MB of stack each up front. Pewter City, Eevee:
  committed memory 7.2-7.4 GB -> 4.7 GB; the memory actually in use is unchanged. Helps on PCs with little RAM or a
  small page file.
- **Less GPU time**: fragment and compute shaders unpack half-precision values natively. RTX 3080, Pewter City at 2x
  resolution: 4.0 -> 3.75 ms of GPU per frame (the frame rate at the game's 30 fps does not change on this machine).
  Frames look the same.
- Fixes from LAS Recomp, the project this port is built on: render passes no longer hand empty color slots to the
  driver (they crash Intel UHD 630 drivers there; not tested here on Intel), a GPU fence race, duplicate pipelines,
  and shader instruction details. A rare crash when two game threads touch new memory at the same moment is fixed
  too (found in another port of the same runtime; not seen in Let's Go).
- After a crash, no extra "Visual C++ Runtime Library" box. A controller Windows does not recognise can be added with
  your own SDL mapping line in `gamecontrollerdb.txt` next to the exe.
- Tested with Eevee and Pikachu 1.0.0 on the regression routes (world, menus, area change); saving from the menu checked with Eevee.
- Upgrading: unzip over your install (no new setup).
- Tested with the zip built for this release, unzipped into a new folder named with accented letters, the game files of 0.3.x installs of both games, started without MSYS2 on the PATH (no recompiling needed): Eevee and Pikachu load a save and reach the world; the start screen rewrote a settings file saved by 0.3.1 with every setting kept.

## 0.3.1 (2026-10-18)

- **Fixes the world breaking after a capture** (found playing 0.3.0 with 60 fps on): after entering and leaving a
  capture, fences, signs and ferns could show flat or dark until you changed area. The cause was older than 0.3.0 and
  only luck kept it hidden: when the game rebuilt a texture in place (it streams them), the texture slot the game had
  registered kept pointing at the freed copy, so what it showed depended on what the memory allocator put there next.
  The render thread and the video-memory models of 0.3.0 changed the allocation pattern, which made it show. Slots
  now follow the texture when the game rebuilds it.
- Tested here with a new unattended loop that walks in the grass of Viridian Forest and runs from each encounter
  (Eevee and Pikachu 1.0.0, both 0.3.0 options on): the world looked the same before and after every encounter. The
  0.3.0 performance options stay **on by default**.
- Upgrading: unzip over your install (no new setup).

## 0.3.0 (2026-10-18)

A performance release. Everything below was measured on one machine (Core i5-14600KF, RTX 3080, Windows 11), with
runs taken in turn (A, B, A, B...) because the same frame varies 10-18 ms of GPU time between runs. **Not measured on
other graphics cards** (AMD, Intel, integrated): feedback is welcome in an issue.

- **The graphics work runs on its own thread (on by default).** The game's calls into the Vulkan backend used to run
  on the game's own thread, in the chain that also runs its logic. Unlimited frame rate on the Pewter City route:
  Eevee 126.9 -> 185.5 fps (frame time ratio 1.46, 95% interval 1.40-1.53), Pikachu 132.1 -> 191.4 fps (1.45, 1.31-1.54);
  the 99th percentile frame time drops from 13.3 to 9.9 ms. At the game's own 30 fps cap you will not see more frames:
  the gain is CPU headroom. The paced 60 fps mode is
  unchanged (16.63 ms per frame with and without). Rendered frames match the old path within the run-to-run noise;
  the Vulkan validation layers report no errors on the regression routes of both games. `render_thread = 0` in
  `larecomp.ini` (or `LARECOMP_RENDER_THREAD=0`) turns it off.
- **Models are read from video memory on dedicated GPUs.** The vertex and index data was read by the GPU in place from
  the game's memory (system memory, over PCIe) for every vertex invocation. It is copied to VRAM now: GPU time of the
  Pewter City frame 10.2 -> 1.2 ms. Integrated GPUs keep the old path (not tested). `vertex_read = direct` in `larecomp.ini` (or
  `LARECOMP_VERTEX_READ=direct`) goes back to it.
- **Mutex waits no longer stall the game threads.** A contended lock spun briefly before sleeping on a condition
  variable (~1000 contended locks per frame per thread, ~12 us each on Windows). The paced 60 fps mode now holds 60.0
  fps (it was 58.8-59.3); unlimited frame rate went from about 60 to 65-74 fps at the time.
- **`larecomp.ini` edited by hand on Windows works**: lines ending in CR+LF kept the CR in the value, so
  `resolution = dynamic`, `upscaler = bilinear` and the percentage modes fell back silently.
- **Dynamic resolution** (`resolution = dynamic`, ini only) judged a steady 30 fps as too slow and dropped to 0.60
  scale; it budgets by the scene's pace now.
- For developers: a much faster regression battery (6 routes of both games in ~100 s, was ~30 min), an interleaved
  A/B performance test with a confidence interval, per-draw GPU timing and a frame time CSV (docs/DEPURACAO.md,
  docs/PESQUISA-GPU.md).
- Upgrading from 0.2.x: unzip over your install; no new setup (the game library is not rebuilt).
- Tested with the zip built for this release, unzipped over 0.2.x installs of both games and started without MSYS2 on the PATH (no recompiling needed): Eevee loads a save and reaches the world (Route 2); Pikachu boots with the render thread on and plays the opening movie.

## 0.2.2 (2026-10-17)

From the second patch in [issue #1](https://github.com/raulzrrs/letsgo-recomp/issues/1), thanks again to
@porstobotgovna:

- **No more crashes on the communication screens.** The network services (`nifm`, `ssl`, `bsd`, `ldn`) didn't exist:
  the game got an empty session and closed when opening local wireless, Mystery Gift or internet co-op. They now
  answer like a console that is offline. Tested on both games (Eevee and Pikachu 1.0.0): **local trade** searches for
  a player and B goes back; **Mystery Gift** stays on "Connecting..." (that screen has no way out offline: the game
  waits with no timeout and ignores B, as on a console without Nintendo's servers); the **Pokémon GO settings**
  search and B goes back to the menu.
- The other Bluetooth (`btm:u`) commands answer with no data, like a console with nothing paired.
- Tested with the zip downloaded from this release, unzipped over 0.2.1 installs of both games (no recompiling
  needed): each one boots a save and reaches the local trade search.

## 0.2.1 (2026-10-17)

From [issue #1](https://github.com/raulzrrs/letsgo-recomp/issues/1), with thanks to @porstobotgovna, who played
*Let's Go, Pikachu!* 1.0.2 on an AMD RX 7800 XT with Windows 10 and sent a patch:

- **Supports *Let's Go, Pikachu!*** (and updated games): the launcher accepts either game by its program id, shows
  which one is set up, keeps Pikachu's saves in `save-pikachu\` and installs translations for the set-up game.
  Tested here with Pikachu 1.0.0: setup (13 to 17 minutes), a new game through the title movie, Professor Oak and the
  name screen, and played (from a copy of an Eevee save) with 60 fps, wild battles and the PT-BR translation. Each
  folder holds one set-up game: to have both, unzip the release twice. **60 fps and wild battles work on Pikachu 1.0.0 too:** their hooks and the generator's instruction patches
  are addresses in the game's executable, now in a table per build (the same functions in Pikachu, found by comparing
  the code instruction by instruction, `tools/addrmap.py`). Other builds (the updates) get neither (greyed out on the
  start screen). The game is told its real version (1.0.0 / 1.0.2) instead of always 1.0.0.
- **Fixes a crash at boot (`c0000005` accessing `0x38`) on some Windows 10 machines**, whose Yu Gothic Regular font
  breaks the game's font engine: Yu Gothic Medium and MS Gothic come first now. `LARECOMP_FONT=<file>` overrides it.
- **Pokémon GO Park settings:** the game waited on a Bluetooth pairing event that was never created and aborted; the
  event exists now, and a wait on an empty handle slot waits on the others.
- **Fixes wild battles with 60 fps on (0.2.0, both games):** the battle counted as a menu and got in-between frames;
  it runs at 30 fps like the game's own battles now.
- Flat dumps (the code files next to the RomFS's `bin/`) are set up too.
- The overlay (close the game?, name entry) speaks English, Portuguese or Russian, from the system language
  (`LARECOMP_LANG=en|pt|ru`).
- Crash reports list the last calls to the host, and the previous run's log is kept as `lasrecomp-prev.log`.
- Tested with the zip downloaded from this release, unzipped twice without development tools on PATH: Eevee 1.0.0
  and Pikachu 1.0.0 set up from `exefs` + `romfs.bin` / `romfs` (13 minutes each), an Eevee save in Viridian City with
  the PT-BR translation and 60 fps in the world, and a new Pikachu game with the translation installed from its `.rar`
  (Professor Oak's lines in Portuguese).

## 0.2.0 (2026-10-17)

Two experimental options on the start screen, both off by default:

- **60 fps (experimental).** The overworld runs at 60 fps natively: the game's own time step is halved there, so
  walking, the camera and the world's animations are smooth at the right speed. Menus and the Pokédex keep the game's
  30 updates per second and are shown at 60 with an in-between frame. Battles and captures stay at 30 fps, as the game
  made them (their timelines count frames; an in-between frame there was tried and left out). Tested by playing:
  world, menus and Pokédex at 60 without speed-ups; battles and captures at normal speed.
- **Wild battles (experimental).** Touching a wild Pokémon starts a classic battle against it (its species, level,
  gender, IVs and level-up moves) and then the capture, like the game does with Snorlax and the legendary birds.
  Played on Route 1. Not done yet: split experience, nature and shiny of the touched Pokémon, regional forms, Pokémon
  with an aura; other areas not tested.

Also:

- **Controller stick dead zone**, adjustable on the start screen ("Keyboard and mouse"): a stick at rest no longer
  moves the Poké Ball aim.
- Known issue, also in 0.1.2: the colored aura of special wild Pokémon sometimes disappears for a single frame
  ([docs/PROGRESSO.md](docs/PROGRESSO.md), item 6).
- Tested with the zip downloaded from this release, in a new folder without development tools on PATH: setup from a
  game folder with `exefs` and `romfs` (13 minutes to recompile on a 20-thread CPU), a new game through the controller
  screen, the title movie, Professor Oak and the character and name screens, and the Brazilian Portuguese fan
  translation installed from its downloaded `.rar` (Professor Oak's lines in Portuguese). With both options on (another
  clean folder): the world at 60 fps and a classic battle started by touching a wild Pidgey.

## 0.1.2 (2026-10-16)

- **Fixes a crash at the title screen on a new game** (no save yet) in 0.1.1: the movie player looked for the
  game's movies in an extracted `romfs` folder, which the release's setup doesn't create (it builds the `romfs.bin`
  image). Movies are now read from inside the image.
- Tested from a clean folder with the downloaded zip: setup, a new game through the controller and language screens
  into the introduction, and the Brazilian Portuguese translation installed from the `.rar`.

## 0.1.1 (2026-10-16)

- **Fixes the game closing right after the controller screen** ("svcBreak") in the 0.1.0 release: its FFmpeg was
  built without the Opus decoder the game's music and sound effects use. The release build now checks every decoder
  the game needs, and a missing one is reported by name instead of crashing.
- The one-time setup's progress bar no longer reaches 100% long before the end (it counted the four game modules as
  two).
- The README states exactly which game files and version are supported, with the sources.

## 0.1.0 (2026-10-15)

First public release.

- Native Windows port of *Pokémon: Let's Go, Eevee!* v1.0.0 by static recompilation (built on
  [LAS Recomp](https://github.com/peiikai/LASRecomp)): the game's four executable modules are translated from ARM64
  to C and compiled to x86-64; no ARM64 interpreter or JIT at run time.
- Release zip with `LetsGoRecomp.exe`: pick your extracted game folder in the launcher and the game code is
  recompiled on your PC (bundled generator and Zig compiler; nothing else to install).
- Stable 30 fps, 720p to 4K internal resolution (with supersampling), SMAA, 16x anisotropic filtering, any window
  size, fullscreen; parallel shader translation with disk caches.
- Keyboard and mouse (rebindable; the mouse aims Poké Balls and is the touch screen), any SDL 3 controller, virtual
  finger with the right stick for Play with Eevee.
- Atmosphere-style LayeredFS mods; one-click install of fan translations from the downloaded .zip / .rar / .7z.
- Launcher in English and Brazilian Portuguese.
