// Ultrawide support (experimental; the launcher's graphics page, g_settings.ultrawide): a screen wider than 16:9 filled
// with the game's image instead of black bars at its sides. The render targets take the window's shape
// (runtime/nvn/output_scale.c, reshaped), the vertex shaders of the draws into them scale the clip x of their
// perspective positions (ShaderKey.widen: the scene, characters and effects alike, whichever matrix carries the
// game's 16:9 projection), the effects read the camera widened the same way (exec.c wide_block), and the interface
// stays in a centred 16:9 box (exec.c, the viewport). Tests: LARECOMP_WIDE=0 / 1 overrides the setting,
// LARECOMP_ASPECT=<width / height> forces an aspect.
//
// (Tried first, in the game: its camera setup 0x70007ebf00, which stores 16/9 next to a field of view, isn't the
// field's camera - never called there - and the 16/9 float in the read-only data, 0x7000f46f80, isn't what its
// projection is made from. Pikachu 1.0.0 addresses.)
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "../hle.h"
#include "../settings.h"

static float g_aspect;  // 0: off (16:9 or narrower, or not asked for)

float widescreen_aspect(void) { return g_aspect; }

// The window's drawable size, every present (runtime/nvn/vulkan/device.c)
void widescreen_window(double w, double h) {
    static int env = -2;
    static float force;
    if (env == -2) {
        env = getenv("LARECOMP_WIDE") ? atoi(getenv("LARECOMP_WIDE")) != 0 : -1;
        force = getenv("LARECOMP_ASPECT") ? (float)atof(getenv("LARECOMP_ASPECT")) : 0;
    }
    bool on = force > 1.0f || (env >= 0 ? env : g_settings.ultrawide);
    float a = !on ? 0 : force > 1.0f ? force : h > 0 ? (float)(w / h) : 0;
    if (a < 16.0f / 9.0f + 0.01f) a = 0;
    if (a != g_aspect) LOG("[wide] aspect %.4f%s\n", a, a ? "" : " (off)");
    g_aspect = a;
}
