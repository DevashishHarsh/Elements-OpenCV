"""Water shield renderer refactor.
One-line parts: helpers, textures, renderer class, standalone water_shield() wrapper.
"""

import cv2
import numpy as np
import time
import math
import random

# Optional MediaPipe (renderer can be embedded into an existing Mediapipe loop)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception:
    mp_hands = None
    mp_drawing = None
    USE_MEDIAPIPE = False

# ---------- Parameters (defaults preserved, simplified to single choices) ----------
CAM_INDEX = 0
DOWNSCALE = 3
ELLIPSE_BASE_SCALE = 4
ELLIPSE_MINOR_FACTOR = 0.72
NOISE_FREQ = 0.2
NOISE_SPEED = 1.5
NOISE_OCTAVES = 1
HIGH_DETAIL_JITTER = 0.06
REFRACT_STRENGTH = 10.0
ALPHA = 0.44
RIM_BRIGHT = 0.75
RIM_NOISE_SCALE = 0.9
SMOOTH_ALPHA = 0.20
MIN_PALM_DIST = 26
SPECULAR_STREAKS = 2
USER_COLOR_RAW = np.array([2, 1.5, 0.0], dtype=np.float32)
if USER_COLOR_RAW.max() <= 0:
    COLOR_TINT = np.array([0.05, 0.35, 1.0], dtype=np.float32)
    COLOR_INTENSITY_SCALE = 1.0
else:
    COLOR_INTENSITY_SCALE = float(USER_COLOR_RAW.max())
    COLOR_TINT = USER_COLOR_RAW / USER_COLOR_RAW.max()
COLOR_INNER = np.clip(np.array([1.0, 1.0, 1.0], dtype=np.float32) * 0.95, 0.0, 1.0)
COLOR_OUTER = np.clip(COLOR_TINT, 0.0, 1.0)
FIST_HOLD_TO_START_CHARGE = 2.0
CHARGE_FADE_DURATION = 1.0
GLOW_COLOR = (30, 35, 0)
GLOW_STRENGTH = 0.45
GLOW_BASE_RADIUS_FACTOR = 0.45

# --------------------------- helpers (kept concise) ---------------------------

def is_palm_open(hand_landmarks, img_w, img_h):
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    extended = []
    thumb_thresh = max(12.0, img_w * 0.03)
    for tip_idx, pip_idx in zip(tips, pips):
        tip = lm[tip_idx]; pip = lm[pip_idx]
        tip_y = tip.y * img_h; pip_y = pip.y * img_h
        tip_x = tip.x * img_w; pip_x = pip.x * img_w
        if tip_idx == 4:
            extended.append(abs(tip_x - pip_x) > thumb_thresh)
        else:
            extended.append((tip_y + 6.0) < pip_y)
    return sum(1 for e in extended if e) >= 4


def count_extended_fingers(hand_landmarks, img_w, img_h):
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    extended = []
    thumb_thresh = max(12.0, img_w * 0.03)
    for tip_idx, pip_idx in zip(tips, pips):
        tip = lm[tip_idx]; pip = lm[pip_idx]
        tip_y = tip.y * img_h; pip_y = pip.y * img_h
        tip_x = tip.x * img_w; pip_x = pip.x * img_w
        if tip_idx == 4:
            extended.append(abs(tip_x - pip_x) > thumb_thresh)
        else:
            extended.append((tip_y + 6.0) < pip_y)
    return sum(1 for e in extended if e)


def is_fist_closed(hand_landmarks, img_w, img_h):
    return count_extended_fingers(hand_landmarks, img_w, img_h) <= 1


def palm_center_and_width(hand_landmarks, w, h):
    idx_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    cx = int((idx_mcp.x + pky_mcp.x + wrist.x) / 3.0 * w)
    cy = int((idx_mcp.y + pky_mcp.y + wrist.y) / 3.0 * h)
    palm_w = max(8.0, math.hypot((idx_mcp.x - pky_mcp.x) * w, (idx_mcp.y - pky_mcp.y) * h))
    return (cx, cy), palm_w


def wrist_to_mid_vector(hand_landmarks, w, h):
    wrist = hand_landmarks.landmark[0]
    mid = hand_landmarks.landmark[12]
    vx = (mid.x - wrist.x) * w
    vy = (mid.y - wrist.y) * h
    ang = math.degrees(math.atan2(vy, vx))
    return np.array([vx, vy], dtype=np.float32), ang


def generate_water_noise(hs, ws, t, freq=NOISE_FREQ, octaves=NOISE_OCTAVES):
    x = np.linspace(-1.0, 1.0, ws, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, hs, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    noise = np.zeros((hs, ws), dtype=np.float32)
    rng = np.random.RandomState(int((t * 1000) % 100000))
    for o in range(octaves):
        freq_o = (1.0 + 0.8 * o) * freq * 5.0
        speed = NOISE_SPEED * (0.6 + 0.5 * o)
        angle = rng.uniform(0, 2 * math.pi)
        kx = math.cos(angle) * freq_o
        ky = math.sin(angle) * freq_o
        phase = t * speed + rng.uniform(-1.0, 1.0)
        amp = 0.7 / (o + 1.0)
        wave = np.sin(xv * kx + yv * ky + phase)
        noise += amp * wave
    adv = np.sin((xv * 3.0 + t * 0.8)) * np.cos((yv * 2.0 + t * 1.2)) * 0.25
    noise += adv
    small_rand = (rng.randn(hs, ws).astype(np.float32) * HIGH_DETAIL_JITTER * 0.6)
    small_rand = cv2.GaussianBlur(small_rand, (7, 7), 0)
    noise += small_rand
    noise -= noise.min()
    if noise.max() > 1e-8:
        noise /= (noise.max() + 1e-8)
    else:
        noise[:] = 0.5
    return noise


def create_color_texture(noise, inner_color=COLOR_INNER, outer_color=COLOR_OUTER, intensity_scale=1.0):
    h, w = noise.shape
    xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(xv * xv + yv * yv)
    r = np.clip(1.0 - r, 0.0, 1.0)
    mix = 0.55 * noise + 0.45 * r
    mix = np.clip(mix, 0.0, 1.0)
    tex = inner_color.reshape(1, 1, 3) * mix[..., None] + outer_color.reshape(1, 1, 3) * (1.0 - mix[..., None])
    tex = tex * float(intensity_scale)
    tex = np.clip(tex, 0.0, 1.0)
    tex_u8 = (tex * 255.0).astype(np.uint8)
    return tex_u8


def apply_refraction_to_crop(crop, noise, refr_strength=REFRACT_STRENGTH):
    ch, cw = crop.shape[:2]
    gx = cv2.Sobel((noise * 255.0).astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel((noise * 255.0).astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    nx = gx / mag
    ny = gy / mag
    disp_x = (nx * (noise - 0.5) * refr_strength).astype(np.float32)
    disp_y = (ny * (noise - 0.5) * refr_strength).astype(np.float32)
    xs = np.tile(np.arange(cw, dtype=np.float32), (ch, 1))
    ys = np.tile(np.arange(ch, dtype=np.float32)[:, None], (1, cw))
    map_x = (xs + disp_x).astype(np.float32)
    map_y = (ys + disp_y).astype(np.float32)
    np.clip(map_x, 0, cw - 1, out=map_x)
    np.clip(map_y, 0, ch - 1, out=map_y)
    warped = cv2.remap(crop, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def make_rim_noise(mask_crop, scale=RIM_NOISE_SCALE):
    h, w = mask_crop.shape
    u8 = (mask_crop * 255.0).astype(np.uint8)
    kernel = max(3, int(min(w, h) * 0.03))
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    eroded = cv2.erode(u8, np.ones((kernel, kernel), np.uint8))
    rim = cv2.subtract(u8, eroded).astype(np.float32) / 255.0
    noise = (np.random.rand(h, w).astype(np.float32) - 0.5) * 2.0
    rim = np.clip(rim + noise * (0.8 * scale), 0.0, 1.0)
    rim = cv2.GaussianBlur(rim, (max(3, kernel), max(3, kernel)), 0)
    rim = np.clip(rim, 0.0, 1.0)
    return rim


def oriented_ellipse_region(frame, center, axes, angle_deg):
    (cx, cy) = center
    (ax, ay) = axes
    r = int(math.hypot(ax, ay) * 1.05)
    x1 = max(0, cx - r); x2 = min(frame.shape[1], cx + r)
    y1 = max(0, cy - r); y2 = min(frame.shape[0], cy + r)
    return x1, y1, x2, y2


def draw_hand_glow(frame, center, radius, color=(120, 170, 255), strength=0.65, fade=1.0):
    h, w = frame.shape[:2]
    glow = np.zeros_like(frame, dtype=np.uint8)
    cv2.circle(glow, (int(center[0]), int(center[1])), max(1, int(radius)), color, -1, cv2.LINE_AA)
    k = max(1, int(radius * 0.9) | 1)
    if k % 2 == 0: k += 1
    glow = cv2.GaussianBlur(glow, (k, k), 0)
    out = frame.astype(np.float32) + (glow.astype(np.float32) * float(strength) * float(fade))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ---------------- Renderer class ----------------
class WaterShieldRenderer:
    """Stateful renderer: call process_frame(frame, hand_landmarks=..., hand_handedness=...) per frame.
    Only triggers on RIGHT hand. Optimized idle path to preserve FPS when no hand is present."""

    def __init__(self, use_mediapipe=USE_MEDIAPIPE):
        self.use_mediapipe = use_mediapipe and (mp_hands is not None)
        self._local_hands = None
        if self.use_mediapipe:
            try:
                self._local_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.7)
            except Exception:
                self._local_hands = None
        # state
        self.fist_first_seen = 0.0
        self.charging_active = False
        self.charging_start_time = 0.0
        self.charging_fade = 0.0
        self.charged = False
        self.shield_visible = False
        self.fade = 0.0
        self._vanish_lock = None
        # performance
        self._frame_counter = 0
        self._fallback_check_interval = 6
        self._idle_skip_time = 0.35

    def process_frame(self, frame, hand_landmarks=None, hand_handedness=None, now=None):
        """Apply shield overlay to 'frame'. Return processed BGR uint8 frame.
        Pass hand_landmarks (mediapipe) and hand_handedness ('Left'/'Right') if available.
        """
        if now is None:
            now = time.time()
        self._frame_counter += 1
        recent_seen = (now - self.fist_first_seen) < self._idle_skip_time if self.fist_first_seen > 0 else False
        if (not recent_seen) and (not self.charging_active) and (not self.charged) and (not self.shield_visible) and hand_landmarks is None:
            if (self._frame_counter % self._fallback_check_interval) != 0:
                return frame
        h, w = frame.shape[:2]
        detected_hand = False; palm_open = False; fist_now = False; center = None; angle = 0.0; major = 0; minor = 0; palm_w = 0

        # choose only RIGHT hand if present
        if hand_landmarks is not None:
            is_right = (hand_handedness is None) or (str(hand_handedness).lower() == 'right')
            if is_right:
                detected_hand = True
                palm_open = is_palm_open(hand_landmarks, w, h)
                fist_now = is_fist_closed(hand_landmarks, w, h)
                (cx, cy), palm_w = palm_center_and_width(hand_landmarks, w, h)
                center = (cx, cy)
                major = int(palm_w * ELLIPSE_BASE_SCALE)
                minor = int(major * ELLIPSE_MINOR_FACTOR)
                vec, angle = wrist_to_mid_vector(hand_landmarks, w, h)
        else:
            # if no landmarks passed, fast return (idle) or allow host to pass landmarks
            detected_hand = False

        # FIST HOLD -> START CHARGE
        if detected_hand and fist_now:
            if self.fist_first_seen == 0.0:
                self.fist_first_seen = now
            time_held = now - self.fist_first_seen
            if (not self.charging_active) and (not self.charged) and (time_held >= FIST_HOLD_TO_START_CHARGE):
                self.charging_active = True
                self.charging_start_time = now
                self.charging_fade = 0.0
        else:
            if not self.charging_active:
                self.fist_first_seen = 0.0
            if self.charging_active and (now - self.charging_start_time) < CHARGE_FADE_DURATION:
                self.charging_active = False
                self.charging_start_time = 0.0
                self.charging_fade = 0.0

        # charging progression
        if self.charging_active:
            self.charging_fade = min(1.0, (now - self.charging_start_time) / CHARGE_FADE_DURATION)
            if self.charging_fade >= 1.0:
                self.charging_active = False
                self.charged = True
                self.charging_start_time = 0.0
                self.charging_fade = 1.0
        if (not self.charging_active) and (not self.charged):
            self.charging_fade = 0.0

        if not detected_hand:
            self.charging_active = False
            self.charging_start_time = 0.0
            self.charging_fade = 0.0
            self.charged = False
            self.fist_first_seen = 0.0
            self.shield_visible = False

        # Opening after charged -> spawn shield
        if detected_hand and palm_open and self.charged and (not self.shield_visible):
            self.shield_visible = True
            self.charged = False
        if self.shield_visible and (not palm_open):
            self.shield_visible = False

        # fade smoothing
        target = 1.0 if self.shield_visible else 0.0
        self.fade = SMOOTH_ALPHA * target + (1.0 - SMOOTH_ALPHA) * self.fade
        if self.fade < 1e-4:
            self.fade = 0.0

        out = frame.copy()

        # immediate fist glow
        if detected_hand and fist_now and center is not None:
            time_held = now - self.fist_first_seen if self.fist_first_seen > 0 else 0.0
            glow_fade = np.clip(min(1.0, time_held / max(0.001, FIST_HOLD_TO_START_CHARGE)), 0.05, 1.0)
            glow_radius = max(12, 1.5 * GLOW_BASE_RADIUS_FACTOR)
            out = draw_hand_glow(out, center, glow_radius, color=GLOW_COLOR, strength=GLOW_STRENGTH, fade=glow_fade)

        # charging sphere overlay
        if self.charging_fade > 1e-4 and center is not None:
            radius = max(16, int(palm_w * 1.6 * 0.9))
            sphere_intensity = 1.0 * self.charging_fade
            out = draw_hand_glow(out, center, radius, color=(220, 180, 60), strength=0.9, fade=sphere_intensity)
            x1 = max(0, int(center[0] - radius)); x2 = min(w, int(center[0] + radius))
            y1 = max(0, int(center[1] - radius)); y2 = min(h, int(center[1] + radius))
            if x2 > x1 and y2 > y1:
                ch = y2 - y1; cw = x2 - x1
                small_h = max(8, ch // DOWNSCALE); small_w = max(8, cw // DOWNSCALE)
                noise = generate_water_noise(small_h, small_w, now * 1.4, freq=NOISE_FREQ * 1.4, octaves=max(2, NOISE_OCTAVES - 1))
                tex = create_color_texture(noise, inner_color=COLOR_INNER, outer_color=COLOR_OUTER, intensity_scale=0.6 * COLOR_INTENSITY_SCALE)
                tex = cv2.resize(tex, (cw, ch), interpolation=cv2.INTER_LINEAR)
                mask = np.zeros((ch, cw), dtype=np.float32)
                cv2.circle(mask, (cw//2, ch//2), int(min(cw, ch)*0.5), 255, -1, cv2.LINE_AA)
                mask = (mask / 255.0)
                mask = cv2.GaussianBlur(mask, (9, 9), 0)
                mask = np.clip(mask, 0.0, 1.0) * self.charging_fade * 0.45
                crop = out[y1:y2, x1:x2].astype(np.float32)
                blended = (crop * (1.0 - mask[..., None]) + tex.astype(np.float32) * mask[..., None])
                out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # main shield rendering
        if self.fade > 0.001 and center is not None:
            x1, y1, x2, y2 = oriented_ellipse_region(out, center, (major, minor), angle)
            if x2 > x1 and y2 > y1:
                crop = out[y1:y2, x1:x2].copy()
                ch, cw = crop.shape[:2]
                small_h = max(10, ch // DOWNSCALE); small_w = max(10, cw // DOWNSCALE)
                noise = generate_water_noise(small_h, small_w, now, freq=NOISE_FREQ, octaves=NOISE_OCTAVES)
                tex = create_color_texture(noise, inner_color=COLOR_INNER, outer_color=COLOR_OUTER, intensity_scale=COLOR_INTENSITY_SCALE)
                tex = cv2.resize(tex, (cw, ch), interpolation=cv2.INTER_LINEAR)
                noise_f = cv2.resize(noise, (cw, ch), interpolation=cv2.INTER_LINEAR)
                warped = apply_refraction_to_crop(crop, noise_f, refr_strength=REFRACT_STRENGTH)
                tinted = (0.35 * warped.astype(np.float32) + 0.65 * tex.astype(np.float32)).astype(np.uint8)
                mask_full = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask_full, center, (major, minor), int(angle), 0, 360, 255, -1, cv2.LINE_AA)
                mask_crop_u8 = mask_full[y1:y2, x1:x2]
                mask_exact = (mask_crop_u8 > 0)
                mask_crop = (mask_crop_u8.astype(np.float32) / 255.0)
                mask_crop = cv2.GaussianBlur(mask_crop, (19, 19), 0)
                mask_crop = np.clip(mask_crop, 0.0, 1.0)
                alpha_local = (mask_crop * ALPHA * self.fade)[..., None]
                composed_inside = (crop.astype(np.float32) * (1.0 - alpha_local) + tinted.astype(np.float32) * alpha_local).astype(np.uint8)
                composed = crop.copy()
                if mask_exact.any():
                    composed[mask_exact] = composed_inside[mask_exact]
                rim_mask = make_rim_noise(mask_crop, scale=RIM_NOISE_SCALE)
                rim_col_img = np.zeros_like(crop, dtype=np.uint8)
                rim_color = (np.array([255, 255, 255], dtype=np.float32) * 0.9).astype(np.uint8)
                for c in range(3):
                    rim_col_img[..., c] = (rim_mask * rim_color[c] * RIM_BRIGHT * self.fade).astype(np.uint8)
                rim_col_img = cv2.GaussianBlur(rim_col_img, (31, 31), 0)
                if (rim_mask > 0.03).any():
                    rim_bin = np.logical_and(rim_mask > 0.03, mask_exact)
                    if rim_bin.any():
                        tmp = composed.astype(np.float32)
                        tmp[rim_bin] = np.clip(tmp[rim_bin] + rim_col_img[rim_bin] * (RIM_BRIGHT * self.fade), 0, 255)
                        composed = tmp.astype(np.uint8)
                hf = (np.random.randn(ch, cw, 1).astype(np.float32) * 8.0 * (mask_crop[..., None] * 0.6))
                composed = composed.astype(np.float32)
                composed += hf
                composed = np.clip(composed, 0, 255).astype(np.uint8)
                out[y1:y2, x1:x2] = composed

        return out


# ---------------- Standalone wrapper (kept minimal) ----------------
def water_shield():
    renderer = WaterShieldRenderer(use_mediapipe=USE_MEDIAPIPE)
    local_hands = renderer._local_hands
    cap = cv2.VideoCapture(CAM_INDEX)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hand_landmarks = None; handedness = None
        if local_hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = local_hands.process(rgb)
            if res.multi_hand_landmarks:
                chosen = None; chosen_h = None
                for idx, hand in enumerate(res.multi_hand_landmarks):
                    label = res.multi_handedness[idx].classification[0].label if (res.multi_handedness is not None and len(res.multi_handedness) > idx) else None
                    if label is not None and label == 'Right':
                        chosen = hand; chosen_h = 'Right'; break
                    if chosen is None:
                        chosen = hand; chosen_h = label
                hand_landmarks = chosen; handedness = (chosen_h.lower() if chosen_h is not None else None)
        out = renderer.process_frame(frame, hand_landmarks=hand_landmarks, hand_handedness=handedness)
        cv2.imshow("Water Shield", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if renderer._local_hands is not None:
        renderer._local_hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    water_shield()
