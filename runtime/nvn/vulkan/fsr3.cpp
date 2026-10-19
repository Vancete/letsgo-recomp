// AMD FidelityFX Super Resolution 3.1 upscaling (C++: the SDK's Vulkan header is) (FidelityFX SDK 1.1.4: the newest FSR with a Vulkan backend - the FSR
// SDK 2.x, FSR 4, is DirectX 12 only), through its C API (third_party/fidelityfx, MIT) and the signed prebuilt
// amd_fidelityfx_vk.dll, loaded at run time next to the exe: without it the upscaler stays FSR 1 (exec.c).
// exec.c post_process hands it the game image at the render resolution with the scene's depth, the motion vectors
// (camera and per object) and a mask of the interface; the scene was jittered by fsr3_jitter's sequence.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "vk.h"
#include "fsr3.h"
}
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/vk/ffx_api_vk.h"

#ifdef _WIN32
#include "ffx_api/ffx_api_loader.h"
static ffxFunctions fx;
#endif
static int loaded = -1;  // -1 not tried, 0 missing, 1 loaded
static ffxContext ctx;
static uint32_t ctx_rw, ctx_rh, ctx_ow, ctx_oh;

static void fsr3_message(uint32_t type, const wchar_t *msg) {
    char m[512];
    snprintf(m, sizeof m, "%ls", msg);
    LOG("[fsr3] %s: %s\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", m);
}

extern "C" bool fsr3_available(void) {
#ifdef _WIN32
    if (loaded < 0) {
        HMODULE m = LoadLibraryA("amd_fidelityfx_vk.dll");
        loaded = m != NULL;
        if (m) ffxLoadFunctions(&fx, m);
        if (loaded && (!fx.CreateContext || !fx.Dispatch || !fx.Query || !fx.DestroyContext)) loaded = 0;
        LOG("[fsr3] amd_fidelityfx_vk.dll %s\n", loaded ? "loaded" : "not found: FSR 1 instead");
    }
    return loaded == 1;
#else
    return false;
#endif
}

// The backend asks the device for its functions; one this device lacks (the debug-utils names and labels of an
// extension not enabled) came back null and was called: a crash in the DLL. Those get a function doing nothing.
static VkResult VKAPI_CALL fsr3_noop(void) { return VK_SUCCESS; }
static PFN_vkVoidFunction VKAPI_CALL fsr3_device_proc(VkDevice dev, const char *name) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(dev, name);
    size_t n = strlen(name);
    if (!f && n > 3 && !strcmp(name + n - 3, "KHR") && n < 128) {  // (an extension's name for what Vulkan 1.1+ has in its core)
        char core[128];
        snprintf(core, sizeof core, "%.*s", (int)(n - 3), name);
        f = vkGetDeviceProcAddr(dev, core);
    }
    if (!f) {
        LOG("[fsr3] the backend asked for %s: not on this device, a no-op\n", name);
        f = (PFN_vkVoidFunction)fsr3_noop;
    }
    return f;
}

#ifdef _WIN32
static bool ensure_context(uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh) {
    if (ctx && ctx_rw == rw && ctx_rh == rh && ctx_ow == ow && ctx_oh == oh) return true;
    if (ctx) {
        vkDeviceWaitIdle(g_vk.dev);
        fx.DestroyContext(&ctx, NULL);
        ctx = NULL;
    }
    ffxCreateBackendVKDesc be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.vkDevice = g_vk.dev, be.vkPhysicalDevice = g_vk.phys, be.vkDeviceProcAddr = fsr3_device_proc;
    // the game image is gamma-encoded (perceptual): FSR linearises it and writes the output the same way
    ffxCreateContextDescUpscale cd = {};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, cd.header.pNext = &be.header;
    cd.flags = FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    cd.maxRenderSize.width = rw, cd.maxRenderSize.height = rh, cd.maxUpscaleSize.width = ow, cd.maxUpscaleSize.height = oh;
    cd.fpMessage = fsr3_message;
    ffxReturnCode_t r = fx.CreateContext(&ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK) {
        LOG("[fsr3] context %ux%u -> %ux%u failed (%u): FSR 1 instead\n", rw, rh, ow, oh, r);
        ctx = NULL;
        loaded = 0;
        return false;
    }
    ctx_rw = rw, ctx_rh = rh, ctx_ow = ow, ctx_oh = oh;
    LOG("[fsr3] context %ux%u -> %ux%u\n", rw, rh, ow, oh);
    return true;
}
#endif

extern "C" void fsr3_jitter(uint32_t rw, uint32_t ow, uint64_t frame, float *jx, float *jy) {
    *jx = *jy = 0;
#ifdef _WIN32
    if (!fsr3_available() || !rw || !ow) return;
    int32_t phases = 8;
    ffxQueryDescUpscaleGetJitterPhaseCount pq = {};
    pq.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    pq.renderWidth = rw, pq.displayWidth = ow, pq.pOutPhaseCount = &phases;
    fx.Query(NULL, &pq.header);  // (a query without a context: the API's static queries)
    if (phases < 1) phases = 8;
    ffxQueryDescUpscaleGetJitterOffset jq = {};
    jq.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
    jq.index = (int32_t)(frame % (uint64_t)phases), jq.phaseCount = phases, jq.pOutX = jx, jq.pOutY = jy;
    fx.Query(NULL, &jq.header);
#else
    (void)frame;
#endif
}

static FfxApiResource res(VkImage img, uint32_t fmt, uint32_t w, uint32_t h, uint32_t usage) {
    FfxApiResourceDescription d = {};
    d.type = FFX_API_RESOURCE_TYPE_TEXTURE2D, d.format = fmt, d.width = w, d.height = h, d.depth = 1, d.mipCount = 1;
    d.flags = 0, d.usage = usage;
    return ffxApiGetResourceVK((void *)img, d, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);  // (every image here: GENERAL)
}

extern "C" bool fsr3_dispatch(VkCommandBuffer cb, const Fsr3Dispatch *in) {
#ifdef _WIN32
    if (!fsr3_available() || !ensure_context(in->rw, in->rh, in->ow, in->oh)) return false;
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = cb;
    d.color = res(in->color, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, in->rw, in->rh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    d.depth = res(in->depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, in->rw, in->rh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    d.motionVectors = res(in->mv, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, in->mvw, in->mvh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    if (in->mask) d.transparencyAndComposition = res(in->mask, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, in->rw, in->rh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    d.output = res(in->output, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM, in->ow, in->oh, FFX_API_RESOURCE_USAGE_UAV);
    d.jitterOffset.x = in->jx, d.jitterOffset.y = in->jy;
    d.motionVectorScale.x = in->mv_scale_x, d.motionVectorScale.y = in->mv_scale_y;
    d.renderSize.width = in->rw, d.renderSize.height = in->rh;
    d.upscaleSize.width = in->ow, d.upscaleSize.height = in->oh;
    d.enableSharpening = in->sharpness > 0.01f;
    d.sharpness = in->sharpness;
    d.frameTimeDelta = in->frame_ms;
    d.preExposure = 1.0f;
    d.reset = in->reset;
    d.cameraNear = in->near_;
    d.cameraFar = in->far_;
    d.cameraFovAngleVertical = in->fov_y;
    d.viewSpaceToMetersFactor = in->meters_per_unit;
    d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    ffxReturnCode_t r = fx.Dispatch(&ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        static int n;
        if (n++ < 4) LOG("[fsr3] dispatch failed (%u)\n", r);
        return false;
    }
    return true;
#else
    (void)cb, (void)in;
    return false;
#endif
}

extern "C" void fsr3_shutdown(void) {
#ifdef _WIN32
    if (ctx) { vkDeviceWaitIdle(g_vk.dev); fx.DestroyContext(&ctx, NULL); ctx = NULL; }
#endif
}

// ---- the frame generation swapchain
// Its context creates FidelityFX's swapchain object (and the real one under it); destroying the context destroys
// both. A resize makes a new context rather than going through the replacement create / destroy pair.
#ifdef _WIN32
static ffxContext sc_ctx;
static VkSwapchainKHR sc_handle;
static PFN_getLastPresentCountFFXAPI sc_present_count;

static VkQueueInfoFFXAPI queue_info(const Fsr3Queue &q) {
    VkQueueInfoFFXAPI i = {};
    i.queue = q.queue, i.familyIndex = q.family, i.submitFunc = NULL;  // (each queue has one submitting thread)
    return i;
}
#endif

extern "C" bool fsr3_sc_create(const VkSwapchainCreateInfoKHR *ci, const Fsr3Queues *q, VkSwapchainKHR *out, Fsr3SwapchainFns *fns) {
#ifdef _WIN32
    if (!fsr3_available()) return false;
    if (sc_ctx) fsr3_sc_destroy();
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    ffxCreateContextDescFrameGenerationSwapChainVK cd = {};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    cd.physicalDevice = g_vk.phys, cd.device = g_vk.dev, cd.swapchain = &sc, cd.allocator = NULL;
    cd.createInfo = *ci;
    cd.createInfo.oldSwapchain = VK_NULL_HANDLE;
    cd.gameQueue = queue_info(q->game), cd.asyncComputeQueue = queue_info(q->async_compute);
    cd.presentQueue = queue_info(q->present), cd.imageAcquireQueue = queue_info(q->image_acquire);
    ffxReturnCode_t r = fx.CreateContext(&sc_ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK || !sc) {
        LOG("[fsr3] frame generation swapchain failed (%u)\n", r);
        sc_ctx = NULL;
        return false;
    }
    ffxQueryDescSwapchainReplacementFunctionsVK rq = {};
    rq.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    r = fx.Query(&sc_ctx, &rq.header);
    if (r != FFX_API_RETURN_OK || !rq.pOutGetSwapchainImagesKHR || !rq.pOutAcquireNextImageKHR || !rq.pOutQueuePresentKHR) {
        LOG("[fsr3] frame generation swapchain: no replacement functions (%u)\n", r);
        fx.DestroyContext(&sc_ctx, NULL);
        sc_ctx = NULL;
        return false;
    }
    fns->images = rq.pOutGetSwapchainImagesKHR;
    fns->acquire = rq.pOutAcquireNextImageKHR;
    fns->present = rq.pOutQueuePresentKHR;
    sc_handle = sc, sc_present_count = rq.pOutGetLastPresentCountFFXAPI;
    *out = sc;
    LOG("[fsr3] frame generation swapchain %ux%u, %u images (queues: game %u, present %u, acquire %u, compute %u)\n",
        ci->imageExtent.width, ci->imageExtent.height, ci->minImageCount, q->game.family, q->present.family,
        q->image_acquire.family, q->async_compute.family);
    return true;
#else
    (void)ci, (void)q, (void)out, (void)fns;
    return false;
#endif
}

extern "C" uint64_t fsr3_sc_present_count(void) {
#ifdef _WIN32
    return sc_ctx && sc_present_count ? sc_present_count(sc_handle) : 0;
#else
    return 0;
#endif
}

extern "C" void fsr3_sc_wait_presents(void) {
#ifdef _WIN32
    if (!sc_ctx) return;
    ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK;
    fx.Dispatch(&sc_ctx, &d.header);
#endif
}

extern "C" void fsr3_sc_destroy(void) {
#ifdef _WIN32
    if (!sc_ctx) return;
    fsr3_sc_wait_presents();
    vkDeviceWaitIdle(g_vk.dev);
    fsr3_fg_destroy();  // (made again for the new swapchain at the next frame)
    fx.DestroyContext(&sc_ctx, NULL);
    sc_ctx = NULL;
#endif
}

// ---- frame generation
// Callback mode: the swapchain runs the generation itself at the present, into its own command buffer, on the image
// presented. frameID goes up by one per generated-for present; a jump of more (after a frame without generation, or a
// cut) makes it start again without history.
#ifdef _WIN32
static ffxContext fg_ctx;
static uint32_t fg_sw, fg_sh, fg_rw, fg_rh;
static uint64_t fg_id = 1;
static bool fg_was_on;
static LARGE_INTEGER fg_last;

static ffxReturnCode_t fg_generate(ffxDispatchDescFrameGeneration *params, void *user) {
    return fx.Dispatch((ffxContext *)user, &params->header);
}

static bool fg_ensure(uint32_t sw, uint32_t sh, uint32_t rw, uint32_t rh, VkFormat fmt) {
    if (fg_ctx && fg_sw == sw && fg_sh == sh && rw <= fg_rw && rh <= fg_rh) return true;
    if (fg_ctx) fsr3_fg_destroy();
    ffxCreateBackendVKDesc be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.vkDevice = g_vk.dev, be.vkPhysicalDevice = g_vk.phys, be.vkDeviceProcAddr = fsr3_device_proc;
    ffxCreateContextDescFrameGeneration cd = {};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, cd.header.pNext = &be.header;
    cd.flags = 0;  // (depth 0..1 not inverted, finite far plane; vectors at the render size without the jitter; SDR)
    cd.displaySize.width = sw, cd.displaySize.height = sh;
    cd.maxRenderSize.width = rw, cd.maxRenderSize.height = rh;
    cd.backBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
    ffxReturnCode_t r = fx.CreateContext(&fg_ctx, &cd.header, NULL);
    if (r != FFX_API_RETURN_OK) {
        LOG("[fsr3] frame generation context %ux%u (render %ux%u) failed (%u)\n", sw, sh, rw, rh, r);
        fg_ctx = NULL;
        return false;
    }
    fg_sw = sw, fg_sh = sh, fg_rw = rw, fg_rh = rh;
    fg_id += 2;  // (a new context: no history)
    LOG("[fsr3] frame generation context: display %ux%u, render up to %ux%u\n", sw, sh, rw, rh);
    return true;
}
#endif

extern "C" bool fsr3_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat sc_format, uint32_t sw, uint32_t sh, const Fsr3FrameGen *f) {
#ifdef _WIN32
    if (!sc_ctx || !sc) return false;
    if (!f) {  // (once when it stops: the swapchain then presents the frames as they come)
        if (fg_ctx && fg_was_on) {
            ffxConfigureDescFrameGeneration c = {};
            c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            c.swapChain = (void *)sc;
            c.frameGenerationEnabled = false;
            c.frameID = fg_id;
            fx.Configure(&fg_ctx, &c.header);
        }
        fg_was_on = false;
        return false;
    }
    if (!fg_ensure(sw, sh, f->rw, f->rh, sc_format)) return false;
    if (!fg_was_on || f->reset) fg_id += 2;  // (no history: what came before is not this scene's previous frame)
    else fg_id++;
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    float ms = fg_last.QuadPart ? (float)((double)(now.QuadPart - fg_last.QuadPart) * 1000.0 / (double)freq.QuadPart) : 33.3f;
    fg_last = now;
    if (ms <= 0 || ms > 100) ms = 33.3f;

    ffxConfigureDescFrameGeneration c = {};
    c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    c.swapChain = (void *)sc;
    c.frameGenerationCallback = fg_generate;
    c.frameGenerationCallbackUserContext = &fg_ctx;
    c.frameGenerationEnabled = true;
    c.allowAsyncWorkloads = false;
    c.HUDLessColor = res(f->hudless, ffxApiGetSurfaceFormatVK(sc_format), sw, sh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    c.flags = getenv("LARECOMP_FG_DEBUG") ? FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES | FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES : 0;
    c.onlyPresentGenerated = getenv("LARECOMP_FG_ONLY_GENERATED") != NULL;  // (tests: the window shows the generated frames alone)
    c.generationRect.left = f->gx, c.generationRect.top = f->gy;
    c.generationRect.width = (int32_t)f->gw, c.generationRect.height = (int32_t)f->gh;
    c.frameID = fg_id;
    ffxReturnCode_t r = fx.Configure(&fg_ctx, &c.header);
    if (r != FFX_API_RETURN_OK) {
        static int n;
        if (n++ < 4) LOG("[fsr3] frame generation configure failed (%u)\n", r);
        fg_was_on = false;
        return false;
    }

    ffxDispatchDescFrameGenerationPrepare p = {};
    p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    p.frameID = fg_id;
    p.flags = c.flags;
    p.commandList = cb;
    p.renderSize.width = f->rw, p.renderSize.height = f->rh;
    p.jitterOffset.x = f->jx, p.jitterOffset.y = f->jy;
    p.motionVectorScale.x = f->mv_scale_x, p.motionVectorScale.y = f->mv_scale_y;
    p.frameTimeDelta = ms;
    p.cameraNear = f->near_, p.cameraFar = f->far_;
    p.cameraFovAngleVertical = f->fov_y;
    p.viewSpaceToMetersFactor = f->meters_per_unit;
    p.depth = res(f->depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, f->rw, f->rh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    p.motionVectors = res(f->mv, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, f->mvw, f->mvh, FFX_API_RESOURCE_USAGE_READ_ONLY);
    r = fx.Dispatch(&fg_ctx, &p.header);
    if (r != FFX_API_RETURN_OK) {
        static int n;
        if (n++ < 4) LOG("[fsr3] frame generation prepare failed (%u)\n", r);
        fg_was_on = false;
        return false;
    }
    if (!fg_was_on) LOG("[fsr3] frame generation on (frame %llu)\n", (unsigned long long)fg_id);
    fg_was_on = true;
    return true;
#else
    (void)cb, (void)sc, (void)sc_format, (void)sw, (void)sh, (void)f;
    return false;
#endif
}

extern "C" void fsr3_fg_destroy(void) {
#ifdef _WIN32
    if (!fg_ctx) return;
    vkDeviceWaitIdle(g_vk.dev);
    fx.DestroyContext(&fg_ctx, NULL);
    fg_ctx = NULL;
    fg_was_on = false;
#endif
}
