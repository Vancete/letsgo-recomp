// FSR 3.1 upscaling (runtime/nvn/vulkan/fsr3.c): the AMD FidelityFX DLL, loaded at run time.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef struct {
    VkImage color, depth, mv, mask, output;  // color RGBA8 (gamma-encoded), depth R32F, mv RGBA16F, mask RGBA8 (or none)
    uint32_t rw, rh, ow, oh, mvw, mvh;       // render size, output size, the motion vectors' size
    float jx, jy;                            // this frame's jitter, in render pixels (fsr3_jitter)
    float mv_scale_x, mv_scale_y;            // the motion vectors to pixels, previous - current
    float near_, far_, fov_y, meters_per_unit, frame_ms, sharpness;
    bool reset;                              // no history (a cut, the first frame)
} Fsr3Dispatch;

bool fsr3_available(void);
// the jitter sequence for render width rw upscaled to ow: frame's offset in pixels
void fsr3_jitter(uint32_t rw, uint32_t ow, uint64_t frame, float *jx, float *jy);
// records the upscale into cb (outside any rendering); false: not done (use FSR 1)
bool fsr3_dispatch(VkCommandBuffer cb, const Fsr3Dispatch *d);
void fsr3_shutdown(void);

// AMD's frame generation swapchain (setting ffx_swapchain; device.c create_swapchain). Step 1 of frame generation:
// the window's swapchain is FidelityFX's (images of its own, copied to the real swapchain on its present queue) with
// no frame generated yet. Its VkSwapchainKHR is not a driver handle: every call on it goes through fns.
typedef struct { VkQueue queue; uint32_t family; } Fsr3Queue;
typedef struct { Fsr3Queue game, async_compute, present, image_acquire; } Fsr3Queues;
typedef struct {
    PFN_vkGetSwapchainImagesKHR images;
    PFN_vkAcquireNextImageKHR acquire;
    PFN_vkQueuePresentKHR present;
} Fsr3SwapchainFns;
// creates the swapchain described by ci (oldSwapchain ignored: fsr3_sc_destroy the previous one first)
bool fsr3_sc_create(const VkSwapchainCreateInfoKHR *ci, const Fsr3Queues *q, VkSwapchainKHR *out, Fsr3SwapchainFns *fns);
void fsr3_sc_wait_presents(void);  // the presents it still has queued are done (before vkDeviceWaitIdle)
uint64_t fsr3_sc_present_count(void);  // presents made to the real swapchain so far (its presenter thread's and passed-through)
void fsr3_sc_destroy(void);        // the swapchain and its context (the GPU idle)

// Frame generation on that swapchain (setting frame_gen; exec.c vk_fg_present): one call per present, before it.
// The swapchain then interpolates a frame between the last one and this one (callback mode: at its present). The
// interface stays the real frame's: hudless is the window image without it (same size and format as the swapchain's),
// and what differs from it is the interface. The rest are FSR 3.1's inputs of this frame (Fsr3Dispatch).
typedef struct {
    VkImage depth, mv, hudless;
    uint32_t rw, rh, mvw, mvh;
    float jx, jy, mv_scale_x, mv_scale_y, near_, far_, fov_y, meters_per_unit;
    bool reset;
    int32_t gx, gy;  // the game image's place in the window: the area interpolated
    uint32_t gw, gh;
} Fsr3FrameGen;
// f NULL: no frame generated before this present (no FSR 3.1 this frame); records its preparation into cb (outside
// any rendering). False: not done.
bool fsr3_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat sc_format, uint32_t sw, uint32_t sh, const Fsr3FrameGen *f);
void fsr3_fg_destroy(void);
