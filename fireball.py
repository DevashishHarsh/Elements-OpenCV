"""Finger fire animation refactored for embedding.

One-line parts: helpers, frame loader, renderer class, standalone fire() wrapper.
"""

import cv2
import numpy as np
import time
import math
import random
import os

# mediapipe optional
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception:
    mp_hands = None
    mp_drawing = None
    USE_MEDIAPIPE = False

# parameters (defaults preserved)
CAM_INDEX = 0
CAP_W = 640
CAP_H = 480
DOWNSCALE = 10
LIGHT_DIST_UNITS = 1.0
LIGHT_BASE_RADIUS = 90.0
LIGHT_INTENSITY = 10

BLUR_KERNEL_BASE = 17
FOG_BLUR = False

CENTER_COLOR = np.array([0.4, 0.4, 0.4], dtype=np.float32)
# single color preset (orange only)
COLOR_PRESET_ORANGE = np.array([0.2, 0.5, 1.00], dtype=np.float32)

COLOR_SPREAD = 3.0

FALLOFF_POWER = 2.2
ORIENT_STRETCH = 1.45

SMOOTH_ALPHA_POS = 0.42
SMOOTH_ALPHA_DIR = 0.32
SMOOTH_ALPHA_LEN = 0.28

EDGE_BOOST_RADIUS_FACTOR = 0.5
EDGE_BOOST_STRENGTH = 0.2
SHADOW_STRENGTH = 0.2
SHADOW_BLUR_SCALE = 1.4
HIGHLIGHT_BOOST = 1.1

MIN_FINGER_LEN_PX = 8.0
MAX_FINGER_LEN_PX = 400.0

FLICKER_AMPLITUDE = 0.12
FLICKER_FREQ = 3.0

GESTURE_FADE_ALPHA = 0.12

FRAME_DIR = "./frames"
FRAME_EXT = "png"
NUM_FRAMES = 100
STARTUP_END = 30
LOOP_START = 31
FLAME_BLUR_KSIZE = 3
ANIM_BASE_SIZE = 120
ANIM_ALPHA_GAIN = 1.6

BLOOM_RADIUS_MULT = 4.5
BLOOM_RADIUS_MIN, BLOOM_RADIUS_MAX = 0.3, 4.0

# helpers

def find_fingertip_fallback(frame):
    # lightweight mask-based fingertip estimator (avoids heavy contour calls)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 15, 40]); upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None, None, mask
    pts = np.column_stack((xs, ys)).astype(np.float32)  # x,y
    mean = pts.mean(axis=0)
    diffs = pts - mean
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmax(dists))
    fingertip = (float(pts[idx,0]), float(pts[idx,1]))
    # PCA for primary direction
    try:
        cov = np.cov(diffs.T)
        vals, vecs = np.linalg.eig(cov)
        major = vecs[:, np.argmax(vals)]
    except Exception:
        major = np.array([1.0, 0.0])
    # ensure major points toward fingertip
    if np.dot(major, (pts[idx] - mean)) < 0:
        major = -major
    dir2d = major / (np.linalg.norm(major) + 1e-8)
    finger_len = float(np.clip(dists[idx], MIN_FINGER_LEN_PX, 300.0))
    return fingertip, dir2d, mask


def compute_light_map_fast(h, w, light_center, base_radius, center_color, edge_color,
                           intensity=1.0, finger_dir=None, color_spread=1.0):
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    dx = xv - light_center[0]; dy = yv - light_center[1]
    if finger_dir is not None:
        angle = np.arctan2(finger_dir[1], finger_dir[0])
        cosA = np.cos(-angle); sinA = np.sin(-angle)
        xr = cosA * dx - sinA * dy
        yr = sinA * dx + cosA * dy
        xr_st = xr / ORIENT_STRETCH
        dist = np.sqrt(xr_st**2 + yr**2)
    else:
        dist = np.sqrt(dx*dx + dy*dy)
    r = dist / (base_radius + 1e-8)
    r_clamped = np.clip(r, 0.0, 1.0)
    r_mix = np.clip(r_clamped ** (1.0 / (color_spread + 1e-9)), 0.0, 1.0)
    fall = (1.0 - r_clamped) ** FALLOFF_POWER
    color = (center_color.reshape(1,1,3) * (1.0 - r_mix[...,None]) +
             edge_color.reshape(1,1,3) * (r_mix[...,None]))
    light_map = color * fall[...,None] * intensity
    return np.clip(light_map, 0.0, 6.0), dist


def edge_boost_layer(gray, light_center, radius, edge_strength=0.9):
    # Canny computed only within an ROI around the bloom to reduce cost
    h, w = gray.shape
    pad = int(max(8, radius * 1.2))
    cx = int(np.clip(light_center[0], 0, w-1)); cy = int(np.clip(light_center[1], 0, h-1))
    x0 = max(0, cx - pad); x1 = min(w, cx + pad)
    y0 = max(0, cy - pad); y1 = min(h, cy + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros_like(gray), np.zeros_like(gray, dtype=np.uint8)
    edges_roi = cv2.Canny((roi*255).astype(np.uint8), 60, 150)
    edges_full = np.zeros_like(gray, dtype=np.uint8)
    edges_full[y0:y1, x0:x1] = edges_roi
    xs = np.arange(w, dtype=np.float32); ys = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    dist = np.sqrt((xv - light_center[0])**2 + (yv - light_center[1])**2)
    fall = np.clip(1.0 - (dist / (radius + 1e-8)), 0.0, 1.0)
    edges_f = (edges_full.astype(np.float32) / 255.0) * fall * edge_strength
    return edges_f, edges_full


def compute_normals_from_gray(gray):
    gx = cv2.Sobel((gray*255).astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel((gray*255).astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.GaussianBlur(gx, (5,5), 0)
    gy = cv2.GaussianBlur(gy, (5,5), 0)
    nx = -gx; ny = -gy; nz = np.ones_like(nx) * 15.0
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    return nx, ny, nz


def is_index_pointing_mediapipe(hand_landmarks, img_w, img_h):
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
    others = [extended[0], extended[2], extended[3], extended[4]]
    is_index_only = (extended[1] and not any(others))
    return bool(is_index_only), extended


def count_extended_fingers_from_contour(mask):
    # faster finger count using distance-transform peaks (avoids contours)
    if mask is None or np.count_nonzero(mask) == 0:
        return 0
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return 0
    x0 = max(0, xs.min() - 5); x1 = min(mask.shape[1], xs.max() + 6)
    y0 = max(0, ys.min() - 5); y1 = min(mask.shape[0], ys.max() + 6)
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    dt = cv2.distanceTransform((roi>0).astype(np.uint8), cv2.DIST_L2, 3)
    if dt.size == 0:
        return 0
    th = dt.max() * 0.6
    peaks = (dt >= th).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(peaks)
    count = max(0, num_labels - 1)
    return min(5, count)


def load_flame_frames(folder, ext="png", count=100):
    frames = [None] * (count + 1)
    for i in range(1, count+1):
        fname = os.path.join(folder, f"{i:04d}.{ext}")
        if not os.path.isfile(fname):
            continue
        im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 2:
            rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            alpha = (im.astype(np.float32) / 255.0)
        elif im.shape[2] == 3:
            rgb = im.astype(np.float32) / 255.0
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            alpha = np.clip((gray - 0.05) * 1.4, 0.0, 1.0)
        else:
            bgra = im.astype(np.float32) / 255.0
            rgb = bgra[..., :3]
            alpha = bgra[..., 3]
        frames[i] = (rgb, alpha)
    return frames


class FireFlameRenderer:
    """Renderer: call process_frame() per-frame. Detects left hand only for triggering effect.

    Optimizations added: when idle (no gesture & no animation) the renderer does a lightweight
    periodic fallback check instead of expensive per-frame processing, and can return the
    original frame early to preserve FPS.
    """

    def __init__(self, frame_dir=FRAME_DIR, frame_ext=FRAME_EXT, num_frames=NUM_FRAMES, use_mediapipe=USE_MEDIAPIPE):
        self.frame_dir = frame_dir; self.frame_ext = frame_ext; self.num_frames = num_frames
        self.use_mediapipe = use_mediapipe and (mp_hands is not None)
        # state
        self.sm_tip = None; self.sm_dir = None; self.sm_len = None
        self.last_seen = 0.0; self.fps = 30.0
        self.gesture_fade = 0.0
        self.prev_gesture_ok = False
        self.anim_state = "idle"; self.anim_idx = 0
        self.anim_last_time = time.time(); self.anim_fps = 30.0
        self.anim_time_per_frame = 1.0 / max(12.0, self.anim_fps)
        self.color_spread = COLOR_SPREAD
        self.preset_color = COLOR_PRESET_ORANGE
        self.flame_frames = load_flame_frames(self.frame_dir, self.frame_ext, self.num_frames)
        self._local_hands = None
        if self.use_mediapipe:
            try:
                self._local_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.6)
            except Exception:
                self._local_hands = None
        self.BLOOM_RADIUS_MULT = BLOOM_RADIUS_MULT
        # vanish lock
        self._vanish_lock_center = None
        # performance helpers
        self._frame_counter = 0
        self._fallback_check_interval = 6    # run fallback detection only every 6 frames when idle
        self._idle_skip_time = 0.35         # if no sighting for this many seconds, mostly skip heavy work

    def process_frame(self, frame, hand_landmarks=None, hand_handedness=None, fingertip=None, finger_dir=None, finger_len=None, mask=None, gesture_ok=None):
        # early fast path: if totally idle and we've not seen a hand recently, skip heavy processing
        self._frame_counter += 1
        now = time.time()
        recent_seen = (now - self.last_seen) < self._idle_skip_time
        if (not recent_seen) and self.anim_state == 'idle' and hand_landmarks is None and fingertip is None:
            # run a lightweight periodic fallback to detect presence; otherwise return frame fast
            if (self._frame_counter % self._fallback_check_interval) != 0:
                return frame
            # otherwise fall through and run lightweight fallback detection below

        h, w = frame.shape[:2]
        fingertip_local = None; finger_dir_local = None; finger_len_local = None; mask_preview = None
        gesture_ok_local = False

        # use mediapipe landmarks if provided; only LEFT hand triggers effect
        if hand_landmarks is not None:
            if hand_handedness is not None:
                is_left = (hand_handedness.lower() == 'left')
            else:
                is_left = True
            lm = hand_landmarks.landmark
            lt = hand_landmarks.landmark[8]; lp = hand_landmarks.landmark[6]; lm_ = hand_landmarks.landmark[5]
            tip_xy = np.array([lt.x * w, lt.y * h], dtype=np.float32)
            pip_xy = np.array([lp.x * w, lp.y * h], dtype=np.float32)
            mcp_xy = np.array([lm_.x * w, lm_.y * h], dtype=np.float32)
            vec = tip_xy - pip_xy
            if np.linalg.norm(vec) < 10.0:
                vec = tip_xy - mcp_xy
            L = np.linalg.norm(vec) + 1e-8
            dirn = vec / L
            fingertip_local = tip_xy.copy(); finger_dir_local = dirn.copy()
            finger_len_local = float(np.clip(L, MIN_FINGER_LEN_PX, MAX_FINGER_LEN_PX))
            self.last_seen = time.time()
            is_index_only, _ = is_index_pointing_mediapipe(hand_landmarks, w, h)
            gesture_ok_local = bool(is_index_only) and is_left

        elif fingertip is not None and finger_dir is not None:
            fingertip_local = np.array([fingertip[0], fingertip[1]], dtype=np.float32)
            finger_dir_local = np.array([finger_dir[0], finger_dir[1]], dtype=np.float32)
            finger_len_local = float(finger_len) if (finger_len is not None) else 40.0
            gesture_ok_local = bool(gesture_ok) if (gesture_ok is not None) else True
            mask_preview = mask
            self.last_seen = time.time()

        else:
            # fallback detection (run only periodically when idle thanks to early path)
            tip, dir2d, mask_fb = find_fingertip_fallback(frame)
            mask_preview = mask_fb
            if tip is not None:
                fingertip_local = np.array([tip[0], tip[1]], dtype=np.float32)
                finger_dir_local = dir2d.copy()
                ys, xs = np.where(mask_fb>0)
                if len(xs)>0:
                    finger_len_local = float(max(8.0, (ys.max()-ys.min())/2.0))
                else:
                    finger_len_local = 40.0
                self.last_seen = time.time()
                n_ext = count_extended_fingers_from_contour(mask_fb)
                gesture_ok_local = (n_ext == 1)

        # smoothing (don't update smoothed position while vanishing)
        if fingertip_local is not None and (self.anim_state != 'vanish'):
            if self.sm_tip is None:
                self.sm_tip = fingertip_local.copy()
            else:
                self.sm_tip = SMOOTH_ALPHA_POS * fingertip_local + (1.0 - SMOOTH_ALPHA_POS) * self.sm_tip
            if finger_dir_local is not None:
                if self.sm_dir is None:
                    self.sm_dir = finger_dir_local.copy()
                else:
                    newd = SMOOTH_ALPHA_DIR * finger_dir_local + (1.0 - SMOOTH_ALPHA_DIR) * self.sm_dir
                    self.sm_dir = newd / (np.linalg.norm(newd) + 1e-8)
            if finger_len_local is not None:
                if self.sm_len is None:
                    self.sm_len = float(finger_len_local)
                else:
                    self.sm_len = SMOOTH_ALPHA_LEN * float(finger_len_local) + (1.0 - SMOOTH_ALPHA_LEN) * self.sm_len

        # gesture fade
        target = 1.0 if gesture_ok_local else 0.0
        self.gesture_fade = GESTURE_FADE_ALPHA * target + (1.0 - GESTURE_FADE_ALPHA) * self.gesture_fade
        if self.gesture_fade < 0.01:
            self.gesture_fade = 0.0

        # detect edges for animation control
        if gesture_ok_local and not self.prev_gesture_ok:
            self.anim_state = "startup"
            self.anim_idx = 1
            self.anim_last_time = time.time()
            self._vanish_lock_center = None
        if (not gesture_ok_local) and self.prev_gesture_ok:
            self.anim_state = "vanish"
            self.anim_idx = STARTUP_END
            self.anim_time_per_frame = 0.8 / float(max(1, STARTUP_END))
            if self.sm_tip is not None and self.sm_dir is not None and self.sm_len is not None:
                offset = self.sm_dir * (LIGHT_DIST_UNITS * self.sm_len)
                self._vanish_lock_center = (self.sm_tip + offset).copy()
            self.anim_last_time = time.time()
        self.prev_gesture_ok = gesture_ok_local

        # animate frames
        now = time.time()
        if self.anim_state != "idle" and (now - self.anim_last_time) >= self.anim_time_per_frame:
            if self.anim_state == "startup":
                self.anim_idx += 1
                if self.anim_idx > STARTUP_END:
                    self.anim_state = "loop"
                    self.anim_idx = LOOP_START if LOOP_START <= self.num_frames else STARTUP_END
                    self.anim_time_per_frame = 1.0 / max(12.0, self.anim_fps)
            elif self.anim_state == "loop":
                self.anim_idx += 1
                if self.anim_idx > self.num_frames:
                    self.anim_idx = LOOP_START
            elif self.anim_state == "vanish":
                self.anim_idx -= 1
                if self.anim_idx <= 0:
                    self.anim_state = "idle"
                    self.anim_idx = 0
                    self._vanish_lock_center = None
            self.anim_last_time = now

        # compute light center (use vanish lock if vanishing)
        if self.anim_state == 'vanish' and (self._vanish_lock_center is not None):
            light_center_full = self._vanish_lock_center
        else:
            if self.sm_tip is not None and self.sm_dir is not None and self.sm_len is not None:
                offset = self.sm_dir * (LIGHT_DIST_UNITS * self.sm_len)
                light_center_full = self.sm_tip + offset
            elif self.sm_tip is not None:
                light_center_full = self.sm_tip + np.array([0.0, -LIGHT_DIST_UNITS * 30.0])
            else:
                light_center_full = np.array([w*0.5, h*0.5], dtype=np.float32)

        base_radius = max(18.0, LIGHT_BASE_RADIUS * max(0.5, (self.sm_len / 60.0 if self.sm_len is not None else 1.0))) * self.BLOOM_RADIUS_MULT

        flicker = 1.0 + math.sin(now * FLICKER_FREQ * 2.0 * math.pi) * (FLICKER_AMPLITUDE*0.6) + (random.random() - 0.5) * (FLICKER_AMPLITUDE*0.6)
        flicker = max(0.7, min(1.3, flicker))

        # compute light map and visual effects
        small_w = max(8, w // DOWNSCALE); small_h = max(8, h // DOWNSCALE)
        light_center_small = (light_center_full / float(DOWNSCALE)).astype(np.float32)
        base_radius_small = max(4.0, (base_radius / float(DOWNSCALE)))
        light_map_small, dist_small = compute_light_map_fast(small_h, small_w, light_center_small, base_radius_small, CENTER_COLOR, self.preset_color, intensity=1.0 * flicker, finger_dir=(self.sm_dir if self.sm_dir is not None else None), color_spread=self.color_spread)

        if FOG_BLUR:
            k = int(max(3, (BLUR_KERNEL_BASE * max(0.5, base_radius_small/20.0))))
            if k % 2 == 0: k += 1
            try:
                lm_uint = np.clip((light_map_small * 255.0).astype(np.uint8), 0, 255)
                lm_blurred = cv2.GaussianBlur(lm_uint, (k,k), 0)
                light_map_small = (lm_blurred.astype(np.float32) / 255.0)
            except Exception:
                pass

        light_map_small *= self.gesture_fade
        light_map = cv2.resize(light_map_small, (w, h), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edges_f, raw_edges = edge_boost_layer(gray, light_center_full, base_radius * EDGE_BOOST_RADIUS_FACTOR, EDGE_BOOST_STRENGTH)

        nx, ny, nz = compute_normals_from_gray(gray)
        xs = np.arange(w, dtype=np.float32); ys = np.arange(h, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        lx = (light_center_full[0] - xv); ly = (light_center_full[1] - yv); lz = 180.0
        ld = np.sqrt(lx*lx + ly*ly + lz*lz) + 1e-8
        Lx = lx / ld; Ly = ly / ld; Lz = lz / ld
        dotNL = nx * Lx + ny * Ly + nz * Lz
        dotNL_clamped = np.clip(dotNL, -1.0, 1.0)
        shadow_raw = np.clip(-dotNL_clamped, 0.0, 1.0)
        att = 1.0 / (1.0 + 0.01 * ld + 0.0006 * (ld**2))
        shadow_raw *= att

        if raw_edges is None or np.count_nonzero(raw_edges) == 0:
            edge_prox = np.zeros_like(gray)
        else:
            inv = cv2.bitwise_not(raw_edges)
            dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
            maxd = max(1.0, base_radius * 1.2)
            edge_prox = np.clip(1.0 - (dt / maxd), 0.0, 1.0)

        shadow_mask = shadow_raw * edge_prox
        sh_small = cv2.resize(shadow_mask, (max(8, w//DOWNSCALE), max(8, h//DOWNSCALE)), interpolation=cv2.INTER_LINEAR)
        k_sh = int(max(3, (BLUR_KERNEL_BASE * (base_radius_small/20.0) * SHADOW_BLUR_SCALE)))
        if k_sh % 2 == 0: k_sh += 1
        try:
            sh_uint = np.clip((sh_small * 255.0).astype(np.uint8), 0, 255)
            sh_bl = cv2.GaussianBlur(sh_uint, (k_sh, k_sh), 0)
            shadow_mask = cv2.resize((sh_bl.astype(np.float32) / 255.0), (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
        shadow_mask = np.clip(shadow_mask * SHADOW_STRENGTH * self.gesture_fade, 0.0, 1.0)

        highlight_raw = np.clip(dotNL_clamped, 0.0, 1.0)
        light_dist = np.sqrt((xv - light_center_full[0])**2 + (yv - light_center_full[1])**2)
        prox = np.clip(1.0 - (light_dist / (base_radius + 1e-8)), 0.0, 1.0)
        highlight_mask = highlight_raw * edge_prox * prox * HIGHLIGHT_BOOST
        hi_small = cv2.resize(highlight_mask, (max(8, w//DOWNSCALE), max(8, h//DOWNSCALE)), interpolation=cv2.INTER_LINEAR)
        k_hi = int(max(3, (BLUR_KERNEL_BASE * (base_radius_small/20.0))))
        if k_hi % 2 == 0: k_hi += 1
        try:
            hi_uint = np.clip((hi_small * 255.0).astype(np.uint8), 0, 255)
            hi_bl = cv2.GaussianBlur(hi_uint, (k_hi, k_hi), 0)
            highlight_mask = cv2.resize((hi_bl.astype(np.float32) / 255.0), (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
        r_local = np.clip(light_dist / (base_radius + 1e-8), 0.0, 1.0)
        local_color = (CENTER_COLOR.reshape(1,1,3) * (1.0 - r_local[...,None]) + self.preset_color.reshape(1,1,3) * (r_local[...,None]))
        highlight_layer = (local_color * highlight_mask[...,None]) * self.gesture_fade

        img_f = frame.astype(np.float32) / 255.0
        add_light = img_f * (light_map * LIGHT_INTENSITY * flicker)
        add_edges = img_f * (np.dstack([edges_f, edges_f, edges_f]) * 0.5)
        composed = img_f + add_light + add_edges + highlight_layer * 0.9
        composed = composed * (1.0 - shadow_mask[...,None])

        if self.anim_state != "idle" and self.anim_idx >= 1 and self.anim_idx <= self.num_frames:
            frame_data = self.flame_frames[self.anim_idx]
            if frame_data is not None:
                flame_rgb, flame_alpha = frame_data
                if FLAME_BLUR_KSIZE > 1:
                    k = FLAME_BLUR_KSIZE
                    if k % 2 == 0: k += 1
                    try:
                        f_rgb = (cv2.GaussianBlur((flame_rgb*255.0).astype(np.uint8), (k,k), 0).astype(np.float32)/255.0)
                        f_alpha = (cv2.GaussianBlur((flame_alpha*255.0).astype(np.uint8), (k,k), 0).astype(np.float32)/255.0)
                    except Exception:
                        f_rgb = flame_rgb
                        f_alpha = flame_alpha
                else:
                    f_rgb = flame_rgb; f_alpha = flame_alpha

                size_factor = (self.sm_len / 40.0) if (self.sm_len is not None) else 1.0
                render_w = int(max(32, ANIM_BASE_SIZE * size_factor))
                h_src, w_src = f_rgb.shape[:2]
                aspect = float(h_src) / float(w_src)
                render_h = int(render_w * aspect)

                try:
                    f_rgb_rs = cv2.resize(f_rgb, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
                    f_alpha_rs = cv2.resize(f_alpha, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    f_rgb_rs = f_rgb; f_alpha_rs = f_alpha

                cx = int(np.clip(light_center_full[0], 0, w-1))
                cy = int(np.clip(light_center_full[1], 0, h-1))
                x0 = cx - render_w//2; y0 = cy - render_h//2
                x1 = x0 + render_w; y1 = y0 + render_h

                sx0 = max(0, x0); sy0 = max(0, y0)
                dx0 = sx0 - x0; dy0 = sy0 - y0
                sx1 = min(w, x1); sy1 = min(h, y1)
                dx1 = sx1 - x0; dy1 = sy1 - y0

                if dx1 > dx0 and dy1 > dy0:
                    dest = composed[sy0:sy1, sx0:sx1, :]
                    src_rgb = f_rgb_rs[dy0:dy1, dx0:dx1, :]
                    src_alpha = f_alpha_rs[dy0:dy1, dx0:dx1]
                    src_alpha = np.expand_dims(np.clip(src_alpha * self.gesture_fade * flicker * ANIM_ALPHA_GAIN, 0.0, 1.0), 2)
                    composed[sy0:sy1, sx0:sx1, :] = np.clip(dest + src_rgb * src_alpha, 0.0, 5.0)

        out = np.clip(composed, 0.0, 5.0)
        out = out / (1.0 + np.max(out) - 1.0)
        out_uint = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        return out_uint


def fire():
    renderer = FireFlameRenderer()
    local_hands = renderer._local_hands

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hand_landmarks = None; handedness = None
        if local_hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = local_hands.process(rgb)
            if res.multi_hand_landmarks:
                # choose LEFT hand if present, otherwise keep first
                chosen = None; chosen_h = None
                for idx, hand in enumerate(res.multi_hand_landmarks):
                    label = res.multi_handedness[idx].classification[0].label if (res.multi_handedness is not None and len(res.multi_handedness) > idx) else None
                    if label is not None and label == 'Left':
                        chosen = hand; chosen_h = 'Left'; break
                    if chosen is None:
                        chosen = hand; chosen_h = label
                hand_landmarks = chosen; handedness = (chosen_h.lower() if chosen_h is not None else None)

        out = renderer.process_frame(frame, hand_landmarks=hand_landmarks, hand_handedness=handedness)
        cv2.imshow("Finger Fire", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire()
