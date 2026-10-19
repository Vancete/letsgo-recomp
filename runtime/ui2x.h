// The game's interface textures at 2x (runtime/ui2x.c). Nothing ships with the app: on the start screen the player
// can have them made once from their own game files - every texture of the interface archives (bin/app/*.arc: LYT
// layouts and BNTX textures) is decoded, upscaled 2x by waifu2x-ncnn-vulkan (nihui, MIT; downloaded then, ~35 MB) and
// compressed again into cache/ui2x.pak. While the game runs, a texture whose content matches one of them by its key
// is drawn from the 2x one (runtime/nvn/vulkan/textures.c); the game still draws it with the same coordinates.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    UI2X_IDLE,
    UI2X_DOWNLOADING,     // fetching the upscaler
    UI2X_EXTRACTING,      // reading the interface textures from the game files
    UI2X_UPSCALING,       // waifu2x at work
    UI2X_PACKING,         // compressing them into cache/ui2x.pak
    UI2X_DONE,
    // failures (ui2x_state)
    UI2X_FAIL_DOWNLOAD,   // no connection, or the download didn't match
    UI2X_FAIL_UPSCALER,   // waifu2x didn't run (no Vulkan GPU it can use?) or gave garbage - details in lasrecomp.log
    UI2X_FAIL_FILES,      // game files missing / disk full
};
void ui2x_generate_start(void);  // on a background thread; the start screen polls the state
int ui2x_state(void);
float ui2x_progress(void);       // 0..1 of the current step
bool ui2x_running(void);
int ui2x_count(void);            // textures in cache/ui2x.pak (0: none made yet)

// ---- while the game runs
// A texture's content key: of level 0's block rows, untiled (the same bytes the BNTX file holds).
uint64_t ui2x_key(const uint8_t *p, uint64_t n);
enum { UI2X_BC1, UI2X_BC3, UI2X_BC4, UI2X_BC5 };
typedef struct {
    uint32_t w, h;  // the 2x image (twice the original's)
    int fmt;        // UI2X_BC*
    uint8_t *data;  // block rows (free())
    uint64_t size;
} Ui2xTex;
// The 2x version of a texture of src_w x src_h with that key, when there is one and they are turned on (g_settings.ui2x).
bool ui2x_find(uint64_t key, uint32_t src_w, uint32_t src_h, Ui2xTex *out);
bool ui2x_enabled(void);  // turned on and cache/ui2x.pak there (cheap: checked once)

#ifdef __cplusplus
}
#endif
