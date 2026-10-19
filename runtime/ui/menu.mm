// In-game overlay (Dear ImGui on the Vulkan renderer; input fed from SDL events): the "close the game?" question
// (Esc, F1 or Minus + Plus on a controller), the F12 settings (volume, brightness, contrast, saturation), the name entry the game asks for (runtime/hle/swkbd.c), the FPS counter
// and the controller's finger on the touch screen (runtime/hle/input.c). Everything else is set on the start screen
// (runtime/ui/launcher_lg.mm).
// LARECOMP_MENU_KEY=<SDL scancode name> ("F10", "Insert"...): another key for the F12 settings (Steam takes F12 for
// screenshots by default).
//
// Threads: SDL events arrive on the main thread and are queued; the overlay is laid out and drawn on the render
// thread inside backend_present, after the game image, into the same window image.
#include <SDL3/SDL.h>
#include <math.h>
#include <pthread.h>
#include <atomic>
#include <vector>

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "osk.h"
extern "C" bool vk_imgui_backend_init(void);  // runtime/nvn/vulkan/imgui_glue.cpp

extern "C" {
#include "../hle.h"
#include "../settings.h"
#include "menu.h"
#include "../paths.h"
void input_reset(void);
bool input_finger_cursor(float *x, float *y, bool *down);  // runtime/hle/input.c
}

// ------------------------------------------------------------------ language
// The overlay speaks English, Portuguese or Russian (issue #1), from the system's language; LARECOMP_LANG=en|pt|ru
// chooses. The launcher has its own strings (runtime/ui/launcher_lg.mm).
static int g_mlang;  // 0 English, 1 Portuguese, 2 Russian
static void pick_menu_language(void) {
    const char *e = getenv("LARECOMP_LANG"), *lang = e && *e ? e : "";
    SDL_Locale **l = NULL;
    int n = 0;
    if (!*lang && (l = SDL_GetPreferredLocales(&n)) && n > 0 && l[0]->language) lang = l[0]->language;
    g_mlang = !strncmp(lang, "pt", 2) ? 1 : !strncmp(lang, "ru", 2) ? 2 : 0;
    SDL_free(l);
}
#define M(en, pt, ru) (g_mlang == 2 ? (ru) : g_mlang == 1 ? (pt) : (en))

// ------------------------------------------------------------------ state shared with the main thread
static std::atomic<bool> g_open{false};  // the "close the game?" question is up
static std::atomic<bool> g_cfg{false};   // the F12 settings are up
static int g_cfg_page;                   // 0 the settings, 1 the advanced ones (render thread)
static std::atomic<bool> g_want_fullscreen_apply{false};

struct QEvent {
    enum { KEY, ANALOG, MOUSE_POS, MOUSE_BUTTON, WHEEL, TEXT } kind;
    ImGuiKey key;
    bool down;
    float value, x, y;
    int button;
    char text[32];
};
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static std::vector<QEvent> q_events;
static float g_pixel_density = 1;

static void push(const QEvent &e) {
    pthread_mutex_lock(&q_lock);
    q_events.push_back(e);
    pthread_mutex_unlock(&q_lock);
}

bool menu_is_open(void) { return g_open.load(); }
static void set_open(bool open) {
    if (open == g_open.load()) return;
    g_open.store(open);
    input_reset();  // nothing held while the question had focus leaks into the game
    if (!open) settings_save();
}
static void set_cfg(bool open) {
    if (open == g_cfg.load()) return;
    if (open) set_open(false);
    g_cfg.store(open);
    if (!open) g_cfg_page = 0;
    input_reset();
    if (!open) settings_save();
}
// (LARECOMP_MENU_AT: runtime/nvn/vulkan/device.c; LARECOMP_MENU_PAGE=cfg / adv: the F12 menu's main / advanced page instead)
extern "C" void menu_test_open(void) {
    const char *pg = getenv("LARECOMP_MENU_PAGE");
    if (!pg) { set_open(true); return; }
    g_cfg_page = !strcmp(pg, "adv");
    set_cfg(true);
}

static ImGuiKey key_from_scancode(SDL_Scancode sc) {
    switch (sc) {
    case SDL_SCANCODE_UP: return ImGuiKey_UpArrow;
    case SDL_SCANCODE_DOWN: return ImGuiKey_DownArrow;
    case SDL_SCANCODE_LEFT: return ImGuiKey_LeftArrow;
    case SDL_SCANCODE_RIGHT: return ImGuiKey_RightArrow;
    case SDL_SCANCODE_RETURN: return ImGuiKey_Enter;
    case SDL_SCANCODE_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDL_SCANCODE_SPACE: return ImGuiKey_Space;
    case SDL_SCANCODE_TAB: return ImGuiKey_Tab;
    case SDL_SCANCODE_BACKSPACE: return ImGuiKey_Backspace;
    case SDL_SCANCODE_DELETE: return ImGuiKey_Delete;
    case SDL_SCANCODE_HOME: return ImGuiKey_Home;
    case SDL_SCANCODE_END: return ImGuiKey_End;
    case SDL_SCANCODE_LSHIFT: return ImGuiKey_LeftShift;
    case SDL_SCANCODE_RSHIFT: return ImGuiKey_RightShift;
    case SDL_SCANCODE_LCTRL: return ImGuiKey_LeftCtrl;
    case SDL_SCANCODE_RCTRL: return ImGuiKey_RightCtrl;
    case SDL_SCANCODE_A: return ImGuiKey_A;
    case SDL_SCANCODE_C: return ImGuiKey_C;
    case SDL_SCANCODE_V: return ImGuiKey_V;
    case SDL_SCANCODE_X: return ImGuiKey_X;
    case SDL_SCANCODE_Z: return ImGuiKey_Z;
    default: return ImGuiKey_None;
    }
}

// Nintendo layout: the right face button (A) confirms, the bottom one (B) goes back.
static ImGuiKey key_from_pad(Uint8 b) {
    switch (b) {
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceDown;   // activate
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceRight; // cancel / back
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    case SDL_GAMEPAD_BUTTON_BACK: return ImGuiKey_GamepadBack;
    default: return ImGuiKey_None;
    }
}

// ---- software keyboard (runtime/hle/swkbd.c: the game's name entry) ------------------------------------------
// The game thread waits in menu_keyboard_run (re-presenting the last frame); this dialog takes the text: typed
// directly when the player last used a keyboard, an on-screen keyboard (d-pad + A, B deletes, Start / + OK) when
// they last used a controller.
static std::atomic<int> g_kbd{0};  // 1 open, 2 confirmed, 3 cancelled
static std::atomic<bool> g_last_pad{false};
static char g_kbd_text[96];
static int g_kbd_max = 8;
static bool capturing() { return g_open.load() || g_cfg.load() || g_kbd.load() == 1; }
// The game gets no input while this is true (runtime/hle/input.c): the question, the settings or the name entry is up.
extern "C" bool menu_input_captured(void) { return capturing(); }

static SDL_Scancode cfg_key(void) {
    static SDL_Scancode k = SDL_SCANCODE_UNKNOWN;
    if (k == SDL_SCANCODE_UNKNOWN) {
        const char *e = getenv("LARECOMP_MENU_KEY");
        k = e && *e ? SDL_GetScancodeFromName(e) : SDL_SCANCODE_UNKNOWN;
        if (k == SDL_SCANCODE_UNKNOWN) k = SDL_SCANCODE_F12;
    }
    return k;
}

bool menu_handle_event(const SDL_Event *e) {
    static bool back_down, start_down;
    if (e->type == SDL_EVENT_KEY_DOWN) g_last_pad = false;
    if (e->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) g_last_pad = true;
    bool typing = g_kbd.load() == 1;  // (the name entry: keys are text, not shortcuts)
    if (typing && e->type == SDL_EVENT_KEY_DOWN && !e->key.repeat) {  // Enter confirms, Esc cancels
        if (e->key.scancode == SDL_SCANCODE_RETURN || e->key.scancode == SDL_SCANCODE_KP_ENTER) { g_kbd = 2; return true; }
        if (e->key.scancode == SDL_SCANCODE_ESCAPE) { g_kbd = 3; return true; }
    }
    if (typing && e->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && e->gbutton.button == SDL_GAMEPAD_BUTTON_START) {
        g_kbd = 2;
        return true;
    }
    switch (e->type) {
    case SDL_EVENT_KEY_DOWN:
        if (!e->key.repeat && !typing && (e->key.scancode == cfg_key() || (e->key.scancode == SDL_SCANCODE_ESCAPE && g_cfg.load()))) {
            set_cfg(e->key.scancode == SDL_SCANCODE_ESCAPE ? false : !g_cfg.load());  // (Esc closes the settings)
            return true;
        }
        if (!e->key.repeat && (e->key.scancode == SDL_SCANCODE_F1 ||
                               (e->key.scancode == SDL_SCANCODE_ESCAPE && !g_open.load()))) {
            if (!typing) { set_cfg(false); set_open(!g_open.load()); }  // (Esc with the question up goes to it below: "Não")
            return true;
        }
        if (!e->key.repeat && (e->key.scancode == SDL_SCANCODE_F11 ||
                               (e->key.scancode == SDL_SCANCODE_RETURN && (e->key.mod & (SDL_KMOD_ALT | SDL_KMOD_GUI))))) {
            g_settings.fullscreen = !g_settings.fullscreen;
            g_want_fullscreen_apply.store(true);
            settings_save();
            return true;
        }
        /* fallthrough */
    case SDL_EVENT_KEY_UP: {
        if (!capturing()) return false;
        bool down = e->type == SDL_EVENT_KEY_DOWN;
        if (down && e->key.scancode == SDL_SCANCODE_ESCAPE) {
            QEvent q = {};
            q.kind = QEvent::KEY; q.key = ImGuiKey_Escape; q.down = true; push(q);
            q.down = false; push(q);
            return true;
        }
        ImGuiKey k = key_from_scancode(e->key.scancode);
        if (k != ImGuiKey_None) { QEvent q = {}; q.kind = QEvent::KEY; q.key = k; q.down = down; push(q); }
        return true;
    }
    case SDL_EVENT_TEXT_INPUT:
        if (!capturing()) return false;
        { QEvent q = {}; q.kind = QEvent::TEXT; strlcpy(q.text, e->text.text, sizeof q.text); push(q); }
        return true;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        bool down = e->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        if (e->gbutton.button == SDL_GAMEPAD_BUTTON_BACK) back_down = down;
        if (e->gbutton.button == SDL_GAMEPAD_BUTTON_START) start_down = down;
        if (down && !typing && ((back_down && start_down) || e->gbutton.button == SDL_GAMEPAD_BUTTON_GUIDE)) {
            set_cfg(false);
            set_open(!g_open.load());
            return true;
        }
        if (!capturing()) return false;
        ImGuiKey k = key_from_pad(e->gbutton.button);
        if (k != ImGuiKey_None) { QEvent q = {}; q.kind = QEvent::KEY; q.key = k; q.down = down; push(q); }
        return true;
    }
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        if (!capturing()) return false;
        float v = e->gaxis.value / 32767.0f;
        QEvent q = {};
        q.kind = QEvent::ANALOG;
        switch (e->gaxis.axis) {
        case SDL_GAMEPAD_AXIS_LEFTX:
            q.key = ImGuiKey_GamepadLStickLeft; q.value = v < -0.3f ? -v : 0; push(q);
            q.key = ImGuiKey_GamepadLStickRight; q.value = v > 0.3f ? v : 0; push(q);
            break;
        case SDL_GAMEPAD_AXIS_LEFTY:
            q.key = ImGuiKey_GamepadLStickUp; q.value = v < -0.3f ? -v : 0; push(q);
            q.key = ImGuiKey_GamepadLStickDown; q.value = v > 0.3f ? v : 0; push(q);
            break;
        default: break;
        }
        return true;
    }
    case SDL_EVENT_MOUSE_MOTION:
        if (!capturing()) return false;
        { QEvent q = {}; q.kind = QEvent::MOUSE_POS; q.x = e->motion.x * g_pixel_density; q.y = e->motion.y * g_pixel_density; push(q); }
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP:
        if (!capturing()) return false;
        {
            QEvent q = {};
            q.kind = QEvent::MOUSE_BUTTON;
            q.button = e->button.button == SDL_BUTTON_LEFT ? 0 : e->button.button == SDL_BUTTON_RIGHT ? 1 : 2;
            q.down = e->type == SDL_EVENT_MOUSE_BUTTON_DOWN;
            push(q);
        }
        return true;
    case SDL_EVENT_MOUSE_WHEEL:
        if (!capturing()) return false;
        { QEvent q = {}; q.kind = QEvent::WHEEL; q.x = e->wheel.x; q.y = e->wheel.y; push(q); }
        return true;
    default:
        return false;
    }
}

void menu_apply_window_requests(void *win) {
    SDL_Window *w = (SDL_Window *)win;
    g_pixel_density = SDL_GetWindowPixelDensity(w);
    static bool text_input;
    bool want_text = capturing();
    if (want_text != text_input) {
        if (want_text) SDL_StartTextInput(w);
        else SDL_StopTextInput(w);
        text_input = want_text;
    }
    static bool first = true;
    if (first || g_want_fullscreen_apply.exchange(false)) {
        first = false;
        SDL_SetWindowFullscreen(w, g_settings.fullscreen);
    }
}

// ------------------------------------------------------------------ render thread
static bool g_inited;
static ImGuiContext *g_ctx;  // ours: the start screen has its own context
static ImFont *g_font, *g_font_big;  // the FPS counter and the name entry
static ImFont *g_menu_font;          // the "close the game?" question (Baloo 2, as on the start screen)

static void style_init(void) {
    ImGuiStyle &s = ImGui::GetStyle();
    ImGui::StyleColorsDark();
    s.WindowRounding = 14; s.FrameRounding = 8; s.GrabRounding = 8;
    s.WindowPadding = ImVec2(28, 24); s.FramePadding = ImVec2(14, 8); s.ItemSpacing = ImVec2(14, 12);
    s.WindowBorderSize = 0;
    ImVec4 *c = s.Colors;
    ImVec4 gold(0.93f, 0.76f, 0.33f, 1), green(0.15f, 0.36f, 0.25f, 1);
    c[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.09f, 0.08f, 0.94f);
    c[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.10f, 0.09f, 0.98f);
    c[ImGuiCol_Text] = ImVec4(0.96f, 0.94f, 0.88f, 1);
    c[ImGuiCol_TextDisabled] = ImVec4(0.62f, 0.64f, 0.58f, 1);
    c[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.19f, 0.16f, 1);
    c[ImGuiCol_FrameBgHovered] = green;
    c[ImGuiCol_FrameBgActive] = ImVec4(0.19f, 0.45f, 0.31f, 1);
    c[ImGuiCol_Button] = ImVec4(0.14f, 0.22f, 0.18f, 1);
    c[ImGuiCol_ButtonHovered] = green;
    c[ImGuiCol_ButtonActive] = ImVec4(0.21f, 0.50f, 0.34f, 1);
    c[ImGuiCol_NavCursor] = gold;
}

static void init(void) {
    IMGUI_CHECKVERSION();
    pick_menu_language();
    g_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.IniFilename = NULL;
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    static const ImWchar ranges[] = {0x20, 0xFF, 0x400, 0x45F, 0};  // (Latin-1: Portuguese accents; Cyrillic)
    const char *paths[] = {"C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"};
    for (const char *p : paths) {
        if (FILE *f = fopen(p, "rb")) fclose(f);  // a missing file is an assertion in ImGui, not a NULL return
        else continue;
        g_font = io.Fonts->AddFontFromFileTTF(p, 34, &cfg, ranges);
        g_font_big = io.Fonts->AddFontFromFileTTF(p, 56, &cfg, ranges);
        break;
    }
    char baloo[1024];
    if (rt_resource("Baloo2-SemiBold.ttf", "third_party/fonts/Baloo2/Baloo2-SemiBold.ttf", baloo, sizeof baloo))
        g_menu_font = io.Fonts->AddFontFromFileTTF(baloo, 39, &cfg, ranges);
    if (!g_font) g_font = io.Fonts->AddFontDefault();
    if (!g_font_big) g_font_big = g_font;
    if (!g_menu_font) g_menu_font = g_font;
    style_init();
    vk_imgui_backend_init();
    g_inited = true;
}

static void drain_events(ImGuiIO &io) {
    pthread_mutex_lock(&q_lock);
    std::vector<QEvent> ev;
    ev.swap(q_events);
    pthread_mutex_unlock(&q_lock);
    for (const QEvent &q : ev) {
        switch (q.kind) {
        case QEvent::KEY: io.AddKeyEvent(q.key, q.down); break;
        case QEvent::ANALOG: io.AddKeyAnalogEvent(q.key, q.value > 0.1f, q.value); break;
        case QEvent::MOUSE_POS: io.AddMousePosEvent(q.x, q.y); break;
        case QEvent::MOUSE_BUTTON: io.AddMouseButtonEvent(q.button, q.down); break;
        case QEvent::WHEEL: io.AddMouseWheelEvent(q.x, q.y); break;
        case QEvent::TEXT: io.AddInputCharactersUTF8(q.text); break;
        }
    }
}

// Reboot (runtime/main.c): the game is gone - the question and the name entry close; the ImGui context stays.
RT_ON_RESET(menu_reset) {
    set_open(false);
    set_cfg(false);
    g_kbd = 0;
    osk_cancel();
    pthread_mutex_lock(&q_lock);
    q_events.clear();
    pthread_mutex_unlock(&q_lock);
}

static void keyboard_draw(ImGuiIO &io, float scale) {
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), io.DisplaySize, IM_COL32(0, 0, 0, 120));
    float w = fminf(io.DisplaySize.x * 0.9f, 1100 * scale);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(w, 0), ImGuiCond_Always);
    ImGui::Begin("##kbd", NULL, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushFont(g_font_big);
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", M("Enter a name", "Digite o nome", "Введите имя"));
    ImGui::PopFont();
    static bool was_open, was_pad;
    bool pad = g_last_pad.load(), appearing = !was_open || pad != was_pad;  // (switching device: focus the new one)
    was_open = true, was_pad = pad;
    if (!pad) {  // keyboard: type straight into the field
        if (appearing) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(-1);
        ImGui::PushFont(g_font_big);
        ImGui::InputText("##name", g_kbd_text, (size_t)(g_kbd_max * 4 + 1 < (int)sizeof g_kbd_text ? g_kbd_max * 4 + 1 : sizeof g_kbd_text),
                         ImGuiInputTextFlags_CallbackCharFilter,
                         [](ImGuiInputTextCallbackData *d) -> int { return d->EventChar < 0x20 ? 1 : 0; });
        ImGui::PopFont();
        if ((int)strlen(g_kbd_text) > g_kbd_max) g_kbd_text[g_kbd_max] = 0;
        ImGui::TextDisabled("%s      (%d / %d)", M("Enter  confirms      Esc  cancels", "Enter  confirma      Esc  cancela",
                                                   "Enter  готово      Esc  отмена"),
                            (int)strlen(g_kbd_text), g_kbd_max);
    } else {  // controller: on-screen keyboard (osk.mm)
        int r = osk_keys(g_kbd_text, sizeof g_kbd_text, g_kbd_max, appearing, scale, g_font_big);
        if (r) g_kbd = r > 0 ? 2 : 3;  // (Start / + confirms in menu_handle_event too)
    }
    ImGui::End();
    if (g_kbd.load() != 1) was_open = false;
}

// Game thread (runtime/hle/swkbd.c): show the dialog, re-present the last frame until it's answered.
// Returns 1 confirmed (out = text, UTF-8), 0 cancelled.
extern "C" int menu_keyboard_run(const char *initial, int max_len, char *out, int cap) {
    extern void backend_represent(void);
    snprintf(g_kbd_text, sizeof g_kbd_text, "%s", initial ? initial : "");
    g_kbd_max = max_len > 0 && max_len < (int)sizeof g_kbd_text / 4 ? max_len : 8;
    input_reset();
    g_kbd = 1;
    // (test) LARECOMP_KBD_AUTO=<text>: answer at once
    if (getenv("LARECOMP_KBD_AUTO")) { snprintf(g_kbd_text, sizeof g_kbd_text, "%s", getenv("LARECOMP_KBD_AUTO")); g_kbd = 2; }
    while (g_kbd.load() == 1 && !g_rt_unwind) {  // ~60 Hz (a reboot cancels)
        uint64_t t0 = SDL_GetTicks();
        backend_represent();
        uint64_t dt = SDL_GetTicks() - t0;
        if (dt < 16) SDL_Delay((uint32_t)(16 - dt));
    }
    bool ok = g_kbd.load() == 2 && g_kbd_text[0];
    if (ok) snprintf(out, cap, "%s", g_kbd_text);
    g_kbd = 0;
    input_reset();
    return ok;
}

// "Close the game?" - Não has the focus first (Enter / A by reflex keeps playing); B / Esc answer Não.
static void quit_draw(ImGuiIO &io, float scale) {
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), io.DisplaySize, IM_COL32(0, 0, 0, 110));
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin("##quit", NULL, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushFont(g_mlang == 2 ? g_font : g_menu_font);  // (Baloo has no Cyrillic: Segoe UI for Russian)
    ImGui::TextUnformatted(M("Close the game?", "Deseja fechar o jogo?", "Закрыть игру?"));
    ImGui::TextDisabled("%s", M("Progress since the last save will be lost.", "O progresso desde o último save será perdido.",
                                "Прогресс после последнего сохранения будет потерян."));
    ImGui::Spacing();
    float bw = 220 * scale;
    static bool focus_no;
    if (!focus_no) { ImGui::SetKeyboardFocusHere(1); focus_no = true; }
    if (ImGui::Button(M("Yes", "Sim", "Да"), ImVec2(bw, 0))) {
        settings_save();
        SDL_Event q = {};
        q.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&q);
    }
    ImGui::SameLine();
    bool no = ImGui::Button(M("No", "Não", "Нет"), ImVec2(bw, 0));
    if (ImGui::Button(M("Settings (F12)", "Configurações (F12)", "Настройки (F12)"), ImVec2(bw * 2 + ImGui::GetStyle().ItemSpacing.x, 0))) {
        focus_no = false;
        set_cfg(true);  // (closes the question: the way in from a controller)
    } else if (no || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        set_open(false);
        focus_no = false;
    }
    ImGui::PopFont();
    ImGui::End();
}

// F12: volume and the game image's colour, applied live (runtime/services/audio.c apply_gain, runtime/nvn/vulkan/exec.c
// post_process); saved to larecomp.ini on closing. B / Esc / F12 close.
// A label and the slider beside it, the sliders lined up at x (the widest label of the page)
static bool cfg_row(const char *label, float x) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(x);
    ImGui::SetNextItemWidth(-1);
    return true;
}
static float cfg_label_x(const char *const *labels, int n) {
    float w = 0;
    for (int i = 0; i < n; i++) w = fmaxf(w, ImGui::CalcTextSize(labels[i]).x);
    return w + ImGui::GetStyle().WindowPadding.x + ImGui::GetStyle().ItemSpacing.x * 2;
}

extern "C" bool vk_fg_available(void);  // (runtime/nvn/vulkan/device.c: AMD's swapchain can be made)

// Advanced settings (the "Advanced..." button): the image's colour, TAA / FSR 3.1 sharpness, the motion vectors and
// the debug views (SSAO and GI have quality levels on the main page instead of parameters: settings_apply_levels)
static void cfg_advanced(float scale, bool *close) {
    const char *l_bri = M("Brightness", "Brilho", "Яркость"), *l_con = M("Contrast", "Contraste", "Контраст"),
               *l_sat = M("Saturation", "Saturação", "Насыщенность"), *l_shp = M("Sharpness", "Nitidez", "Резкость");
    const char *labels[] = {l_bri, l_con, l_sat, l_shp};
    float x = cfg_label_x(labels, 4);
    // the image's colour (runtime/nvn/vulkan/exec.c post_process, color.frag)
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", M("Image", "Imagem", "Изображение"));
    {
        struct Row { const char *label; float *v, lo, hi; } rows[] = {
            {l_bri, &g_settings.brightness, 0.5f, 1.5f}, {l_con, &g_settings.contrast, 0.5f, 1.5f}, {l_sat, &g_settings.saturation, 0.0f, 2.0f}};
        for (int i = 0; i < 3; i++) {
            float pct = *rows[i].v * 100;
            cfg_row(rows[i].label, x);
            ImGui::PushID(100 + i);
            if (ImGui::SliderFloat("##c", &pct, rows[i].lo * 100, rows[i].hi * 100, "%.0f%%", ImGuiSliderFlags_AlwaysClamp)) *rows[i].v = pct / 100;
            ImGui::PopID();
            if (!i) ImGui::SetItemDefaultFocus();
        }
    }
    // TAA and the motion vectors (runtime/nvn/vulkan/exec.c taa_run, mv_run; per object: maxwell/translate.c velocity)
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", M("Temporal (TAA / FSR 3.1)", "Temporal (TAA / FSR 3.1)", "Временное (TAA / FSR 3.1)"));
    ImGui::BeginDisabled(g_settings.antialias != AA_TAA && g_settings.antialias != AA_FSR3);  // (FSR 3.1's sharpening too)
    {
        float pct = g_settings.taa_sharpness * 100;
        cfg_row(l_shp, x);
        if (ImGui::SliderFloat("##taa_s", &pct, 0, 100, "%.0f%%", ImGuiSliderFlags_AlwaysClamp)) g_settings.taa_sharpness = pct / 100;
    }
    ImGui::EndDisabled();
    ImGui::Checkbox(M("Per-object motion vectors", "Vetores de movimento por objeto", "Векторы движения объектов"), &g_settings.mv_objects);
    // debug views (shown at the present instead of the game image: exec.c post_process)
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", M("Debug", "Depuração", "Отладка"));
    ImGui::BeginDisabled(!g_settings.ssao);
    ImGui::Checkbox(M("Occlusion alone", "Só a oclusão", "Только затенение"), &g_settings.ssao_debug);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!g_settings.gi);
    ImGui::Checkbox(M("Indirect light alone", "Só a luz indireta", "Только непрямой свет"), &g_settings.gi_debug);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox(M("Motion vectors", "Vetores de movimento", "Векторы движения"), &g_settings.mv_debug);
    settings_clamp_color();
    ImGui::Spacing();
    float bw = 240 * scale;
    if (ImGui::Button(M("Back", "Voltar", "Назад"), ImVec2(bw, 0)) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))
        g_cfg_page = 0;
    ImGui::SameLine();
    if (ImGui::Button(M("Reset", "Restaurar", "Сбросить"), ImVec2(bw, 0))) {
        g_settings.ssao_debug = g_settings.gi_debug = g_settings.mv_debug = false;
        g_settings.taa_sharpness = 0.5f, g_settings.mv_objects = true;
        g_settings.brightness = g_settings.contrast = g_settings.saturation = 1.0f;
    }
    ImGui::SameLine();
    *close = ImGui::Button(M("Close", "Fechar", "Закрыть"), ImVec2(bw, 0));
}

static void cfg_draw(ImGuiIO &io, float scale) {
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), io.DisplaySize, IM_COL32(0, 0, 0, 70));
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    float ww = fminf(io.DisplaySize.x * 0.9f, 900 * scale);
    ImGui::SetNextWindowSize(ImVec2(ww, 0), ImGuiCond_Always);
    // (taller than the screen - the advanced page at 720p: it scrolls, and keyboard / pad navigation follows the focus)
    ImGui::SetNextWindowSizeConstraints(ImVec2(ww, 0), ImVec2(ww, io.DisplaySize.y * 0.92f));
    // (one window per page: a page appearing gets its default focus)
    ImGui::Begin(g_cfg_page ? "##cfg_adv" : "##cfg", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    // (fixed width, height fitted: the sliders fill the line, which an auto-resized window cannot have)
    ImGui::PushFont(g_mlang == 2 ? g_font : g_menu_font);
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", g_cfg_page ? M("Advanced settings", "Configurações avançadas", "Дополнительно")
                                                                       : M("Settings", "Configurações", "Настройки"));
    ImGui::PopFont();
    bool close = false;
    if (g_cfg_page) {
        cfg_advanced(scale, &close);
    } else {
        const char *labels[] = {M("Volume", "Volume", "Громкость"), M("Shadow filter", "Filtro das sombras", "Фильтр теней"),
                                M("Anti-aliasing", "Antisserrilhado", "Сглаживание"), M("Frame generation", "Geração de quadros", "Генерация кадров"),
                                M("FSR 3.1 mode", "Modo do FSR 3.1", "Режим FSR 3.1"),
                                M("Ambient occlusion (SSAO)", "Oclusão de ambiente (SSAO)", "Затенение (SSAO)"),
                                M("Indirect light (GI)", "Luz indireta (GI)", "Непрямой свет (GI)"),
                                M("Frame rate limit", "Limite de quadros", "Лимит кадров")};
        float x = cfg_label_x(labels, 8);
        {
            float pct = g_settings.volume * 100;
            cfg_row(labels[0], x);
            if (ImGui::SliderFloat("##vol", &pct, 0, 200, "%.0f%%", ImGuiSliderFlags_AlwaysClamp)) g_settings.volume = pct / 100;
            ImGui::SetItemDefaultFocus();  // (d-pad / arrows pick a row; A / Enter, then left-right, changes it)
        }
        // the world's frame rate (runtime/hle/fps60.c, switched at the next present): the game's 30, 60, or 120 (more
        // experimental); menus stay at 30 with an in-between image (60), battles and captures at 30
        {
            const char *fr[] = {M("30 fps (original)", "30 fps (original)", "30 fps (оригинал)"), "60 fps",
                                M("120 fps (experimental)", "120 fps (experimental)", "120 fps (экспериментально)")};
            int f = g_settings.fps == 120 ? 2 : g_settings.fps == 60 ? 1 : 0;
            cfg_row(labels[7], x);
            if (ImGui::Combo("##fps", &f, fr, 3)) g_settings.fps = f == 2 ? 120 : f == 1 ? 60 : 30;
        }
        ImGui::Separator();
        // SSAO and GI by quality level (their parameters: settings_apply_levels; high = what they had before the levels)
        const char *lv[] = {M("Off", "Desligado", "Выкл."), M("Low", "Baixo", "Низкое"), M("Medium", "Médio", "Среднее"),
                            M("High", "Alto", "Высокое")};
        cfg_row(labels[5], x);
        ImGui::Combo("##ssao_lv", &g_settings.ssao_level, lv, 4);
        cfg_row(labels[6], x);
        ImGui::Combo("##gi_lv", &g_settings.gi_level, lv, 4);
        // the shadow map's filter (runtime/nvn/vulkan/textures.c vk_sampler, runtime/nvn/maxwell/translate.c shpcf)
        const char *names[] = {M("Game's", "Do jogo", "Как в игре"), M("Bilinear", "Bilinear", "Билинейный"),
                               M("Smooth", "Suave", "Мягкий")};
        cfg_row(labels[1], x);
        ImGui::Combo("##shadow_filter", &g_settings.shadow_filter, names, 3);
        // anti-aliasing: SMAA on the final image (the interface too), or TAA on the scene (jittered, motion vectors:
        // runtime/nvn/vulkan/exec.c taa_run)
        // FSR 3.1 last: temporal (jittered scene, motion vectors, the interface masked: exec.c, fsr3.cpp), it anti-aliases
        // and upscales a lower render resolution (its mode, below). Otherwise a lower resolution is scaled bilinearly.
        const char *aa[] = {M("Off", "Desligado", "Выкл."), "SMAA", "TAA", "FSR 3.1"};
        cfg_row(labels[2], x);
        if (ImGui::Combo("##aa", &g_settings.antialias, aa, 4)) settings_apply_levels();
        // FSR 3.1's presets: the render resolution is the window's divided by the ratio (runtime/nvn/output_scale.c)
        if (g_settings.antialias == AA_FSR3) {
            const char *pr[] = {M("Native AA (1.0x)", "AA nativo (1,0x)", "Нативное AA (1,0x)"),
                                M("Quality (1.5x)", "Qualidade (1,5x)", "Качество (1,5x)"),
                                M("Balanced (1.7x)", "Equilibrado (1,7x)", "Баланс (1,7x)"),
                                M("Performance (2.0x)", "Desempenho (2,0x)", "Производительность (2,0x)"),
                                M("Ultra Performance (3.0x)", "Ultra desempenho (3,0x)", "Макс. производительность (3,0x)")};
            cfg_row(labels[4], x);
            ImGui::Combo("##fsr3p", &g_settings.fsr3_preset, pr, 5);
        }
        // FSR 3.1 frame generation (exec.c vk_fg_present; AMD's swapchain is switched to at the next frame: device.c). FSR
        // 3.1 makes one frame between two (several is FSR 4's, which has no Vulkan version): off or 2x.
        {
            bool can = g_settings.antialias == AA_FSR3 && vk_fg_available();
            const char *fg[] = {M("Off", "Desligado", "Выкл."), "2x"};
            int cur = can && g_settings.frame_gen == 2 ? 1 : 0;
            cfg_row(labels[3], x);
            ImGui::BeginDisabled(!can);
            if (ImGui::Combo("##fg", &cur, fg, 2)) g_settings.frame_gen = cur ? 2 : 0;
            ImGui::EndDisabled();
            if (!can) ImGui::SetItemTooltip("%s", vk_fg_available() ? M("With FSR 3.1 anti-aliasing", "Com o antisserrilhado FSR 3.1",
                                                                        "Со сглаживанием FSR 3.1")
                                                                    : M("amd_fidelityfx_vk.dll is missing", "Falta amd_fidelityfx_vk.dll",
                                                                        "Нет amd_fidelityfx_vk.dll"));
        }
        settings_apply_levels();
        settings_clamp_color();
        ImGui::Spacing();
        float bw = 240 * scale;
        if (ImGui::Button(M("Advanced...", "Avançado...", "Дополнительно..."), ImVec2(bw, 0))) g_cfg_page = 1;
        ImGui::SameLine();
        if (ImGui::Button(M("Reset", "Restaurar", "Сбросить"), ImVec2(bw, 0))) g_settings.volume = 1.0f;
        ImGui::SameLine();
        close = ImGui::Button(M("Close", "Fechar", "Закрыть"), ImVec2(bw, 0)) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false);
    }
    ImGui::TextDisabled("%s", M("F12 or Esc closes", "F12 ou Esc fecha", "F12 или Esc — закрыть"));
    ImGui::End();
    if (close) set_cfg(false);
}

void menu_render(void *cmd, uint32_t target_w, uint32_t target_h) {
    bool open = g_open.load(), cfg = g_cfg.load(), kbd = g_kbd.load() == 1;
    float fx, fy;
    bool fdown, finger = !open && !cfg && input_finger_cursor(&fx, &fy, &fdown);
    if (!open && !cfg && !kbd && !g_settings.show_fps && !finger) {
        if (g_inited) { pthread_mutex_lock(&q_lock); q_events.clear(); pthread_mutex_unlock(&q_lock); }
        return;
    }
    if (!g_inited) init();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)target_w, (float)target_h);
    io.DisplayFramebufferScale = ImVec2(1, 1);
    static double last;
    double now = SDL_GetTicks() / 1000.0;
    io.DeltaTime = last > 0 && now > last ? (float)(now - last) : 1.0f / 60;
    last = now;
    drain_events(io);
    // fonts are built for a ~1080 px tall output; scale to the window (spacing / rounding too)
    float scale = (float)(target_h / 1080.0);
    if (scale < 0.45f) scale = 0.45f;
    io.FontGlobalScale = scale;
    static ImGuiStyle base;
    static float style_scale = -1;
    if (style_scale < 0) base = ImGui::GetStyle();
    if (fabsf(style_scale - scale) > 0.01f) {
        ImGui::GetStyle() = base;
        ImGui::GetStyle().ScaleAllSizes(scale);
        style_scale = scale;
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    if (g_settings.show_fps) {
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 16 * scale, 16 * scale), 0, ImVec2(1, 0));
        ImGui::SetNextWindowBgAlpha(0.5f);
        ImGui::Begin("##fps", NULL, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::Text("%.0f fps", io.Framerate);
        ImGui::End();
    }
    if (finger) {  // the controller's finger on the touch screen (right stick moves it, ZR touches)
        ImDrawList *dl = ImGui::GetForegroundDrawList();
        float r = 26 * scale;
        dl->AddCircleFilled(ImVec2(fx, fy), r, fdown ? IM_COL32(255, 255, 255, 170) : IM_COL32(255, 255, 255, 60));
        dl->AddCircle(ImVec2(fx, fy), r, IM_COL32(255, 255, 255, 235), 0, 4 * scale);
    }
    if (open) quit_draw(io, scale);
    else if (cfg && !kbd) cfg_draw(io, scale);
    if (kbd) keyboard_draw(io, scale);
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), (VkCommandBuffer)cmd);  // inside the window image's rendering
}
