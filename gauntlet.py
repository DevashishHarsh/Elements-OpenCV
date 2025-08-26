"""
Infinity Gauntlet renderer (refactored)
- One-line labels only, no debug overlays, importable class with process_frame
- Adds yellow hand glow while stones present; glow lasts HAND_GLOW_DURATION seconds and enables snap only during glow
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math

# --- config ---
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480
FLIP = False

FIST_HOLD_SECONDS = 2.0
STONE_FADE_IN = 0.45
STONE_FADE_OUT = 0.6

STONE_COLORS = [
    (0, 140, 255),
    (0, 0, 255),
    (255, 0, 0),
    (204, 0, 255),
    (0, 220, 30),
    (0, 255, 255)
]
STONE_LM_IDXS = [17, 13, 9, 5, 2, None]
STONE_GLOW_STEPS = 6

TOUCH_FRAC = 0.10
RELEASE_FRAC = 0.26
VEL_THRESH_FRAC = 2.2
MIN_HOLD_TIME = 0.06
COOLDOWN_SEC = 0.9

DISINTEGRATE_FRAMES = 45
PARTICLE_SIZE = 12
PARTICLE_SPREAD = 18

YELLOW_FLASH_DURATION = 5

# Hand glow configuration
HAND_GLOW_DURATION = 10.0  # seconds that the hand glow remains (enables snap during this time)
HAND_GLOW_COLOR = (70, 70, 70)  # BGR yellow-ish
HAND_GLOW_STRENGTH = 0.85
HAND_GLOW_BLUR_SCALE = 0.9

# --- mediapipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- helpers (one-line each) ---
def lm_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def get_orientation(hand_landmarks, handedness_label):
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z])
    index_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z])
    pinky_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z])
    v1 = index_mcp - wrist; v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    if handedness_label == "Right":
        normal = -normal
    return "Front" if normal[2] < 0 else "Back"

def is_fist_by_landmarks(landmarks, w, h):
    idx_mcp = lm_xy(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], w, h)
    pky_mcp = lm_xy(landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], w, h)
    palm_w = max(1.0, dist(idx_mcp, pky_mcp))
    folded = 0
    tip_idx = [8, 12, 16, 20]
    pip_idx = [6, 10, 14, 18]
    for t, p in zip(tip_idx, pip_idx):
        tip = lm_xy(landmarks.landmark[t], w, h)
        pip = lm_xy(landmarks.landmark[p], w, h)
        if dist(tip, pip) < 0.45 * palm_w:
            folded += 1
    return folded >= 3

def compute_back_of_hand_center(landmarks, w, h):
    pts = [lm_xy(landmarks.landmark[i], w, h) for i in (5,9,13,17)]
    center = np.mean(pts, axis=0)
    wrist = lm_xy(landmarks.landmark[0], w, h)
    nudge = (wrist - center) * 0.22
    return center + nudge

def wrist_to_mid_vector(hand_landmarks, w, h):
    wrist = hand_landmarks.landmark[0]
    mid = hand_landmarks.landmark[12]
    vx = (mid.x - wrist.x) * w
    vy = (mid.y - wrist.y) * h
    ang = math.degrees(math.atan2(vy, vx))
    return np.array([vx, vy], dtype=np.float32), ang

def draw_simple_gradient_ellipse(img, center, rx, ry, color_bgr, alpha):
    if alpha <= 0.01:
        return img
    cx, cy = int(center[0]), int(center[1])
    maxr = max(1, int(max(rx, ry)))
    b, g, r = color_bgr
    dark = (int(b*0.55), int(g*0.55), int(r*0.55))
    steps = 18
    angle = 0
    for i in range(steps, 0, -1):
        t = i / float(steps)
        if t > 0.6:
            tt = (t - 0.6) / 0.4
            cb = int(dark[0] * tt + b * (1-tt))
            cg = int(dark[1] * tt + g * (1-tt))
            cr = int(dark[2] * tt + r * (1-tt))
        else:
            tt = t / 0.6
            cb = int(b * tt + 255 * (1-tt))
            cg = int(g * tt + 255 * (1-tt))
            cr = int(r * tt + 255 * (1-tt))
        color = (cb, cg, cr)
        ar = alpha * (0.9 * (1.0 - (i / float(steps)) * 0.6))
        axes = (int(rx * t), int(ry * t))
        if axes[0] < 1 or axes[1] < 1:
            continue
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, ar, img, 1 - ar, 0)
    return img

# --- Stone class ---
class Stone:
    def __init__(self, color_bgr, lm_idx):
        self.color = color_bgr
        self.lm_idx = lm_idx
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.alpha = 0.0
        self.state = "off"
        self.spawn_t = 0.0
    def update(self, landmarks, w, h, back_center=None):
        if self.lm_idx is None:
            if back_center is not None:
                self.pos = back_center.copy()
        else:
            lm = landmarks.landmark[self.lm_idx]
            self.pos = lm_xy(lm, w, h)

# --- Renderer class ---
class InfinityGauntletRenderer:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    model_complexity=0,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.7) if use_mediapipe else None
        self.stones = [Stone(STONE_COLORS[i], STONE_LM_IDXS[i]) for i in range(6)]
        self.prev_dist = None
        self.prev_time = None
        self.ending = False
        self.armed = False
        self.armed_since = 0.0
        self.last_snap_time = -999.0
        self.disintegration_active = False
        self.disintegration_counter = 0
        self.disintegration_particles = None
        self.white_after_disintegration = False
        self.white_show_start = None
        self.fist_start_time = None
        self.stones_present = False
        self.yellow_flash_active = False
        self.yellow_flash_start = 0.0
        # hand glow state
        self.hand_glow_active = False
        self.hand_glow_start = 0.0
        self.snap_allowed = False

    def _render_hand_glow(self, out, landmarks, palm_w_est, now):
        # build convex hull mask from all 21 landmarks
        h, w = out.shape[:2]
        pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark], dtype=np.int32)
        try:
            hull = cv2.convexHull(pts)
        except Exception:
            hull = pts
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            cv2.fillConvexPoly(mask, hull, 255)
        except Exception:
            cv2.fillPoly(mask, [pts], 255)

        # blur mask to produce soft glow
        k = int(max(3, round(palm_w_est * HAND_GLOW_BLUR_SCALE)))
        if k % 2 == 0:
            k += 1
        mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
        mask_f = (mask_blur.astype(np.float32) / 255.0) * HAND_GLOW_STRENGTH

        # color overlay (yellow) and soft additive composite
        color_layer = np.zeros_like(out, dtype=np.float32)
        color_layer[:] = np.array(HAND_GLOW_COLOR, dtype=np.float32)
        overlay = (color_layer * mask_f[..., None])
        try:
            overlay_blur = cv2.GaussianBlur(overlay.astype(np.uint8), (k,k), 0).astype(np.float32)
            out_f = out.astype(np.float32)
            out_f = np.clip(out_f + overlay_blur * 1.0, 0, 255)
            out[:] = out_f.astype(np.uint8)
        except Exception:
            out_f = out.astype(np.float32)
            out_f = np.clip(out_f + overlay * 0.9, 0, 255)
            out[:] = out_f.astype(np.uint8)
        return out

    def process_frame(self, frame, hand_landmarks=None, handedness_label=None, now=None):
        """Process one BGR frame; return processed BGR frame."""
        if now is None:
            now = time.time()
        if FLIP:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # handle disintegration animation if active (unchanged)
        if self.disintegration_active:
            if self.disintegration_particles is None:
                particles = []
                for y in range(0, h, PARTICLE_SIZE):
                    for x in range(0, w, PARTICLE_SIZE):
                        y1 = min(y + PARTICLE_SIZE, h)
                        x1 = min(x + PARTICLE_SIZE, w)
                        patch = frame[y:y1, x:x1].copy()
                        ph, pw = patch.shape[:2]
                        if ph == 0 or pw == 0:
                            continue
                        if patch.mean() < 8:
                            continue
                        particles.append({
                            "pos": np.array([x + random.uniform(0, pw), y + random.uniform(0, ph)], dtype=np.float32),
                            "color": patch,
                            "size": (ph, pw),
                            "vel": np.array([random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD),
                                             random.uniform(-PARTICLE_SPREAD*0.7, PARTICLE_SPREAD*0.3)], dtype=np.float32)
                        })
                self.disintegration_particles = particles
            dust_frame = np.zeros_like(frame)
            for p in self.disintegration_particles:
                p["pos"] += p["vel"] * 0.28
                x = int(p["pos"][0]); y = int(p["pos"][1])
                ph, pw = p["size"]
                dst_x0 = max(0, x); dst_y0 = max(0, y)
                dst_x1 = min(w, x + pw); dst_y1 = min(h, y + ph)
                sx0 = 0 if x >= 0 else -x
                sy0 = 0 if y >= 0 else -y
                sx1 = sx0 + (dst_x1 - dst_x0)
                sy1 = sy0 + (dst_y1 - dst_y0)
                if dst_x1 > dst_x0 and dst_y1 > dst_y0 and sx1 > sx0 and sy1 > sy0:
                    dust_frame[dst_y0:dst_y1, dst_x0:dst_x1] = p["color"][sy0:sy1, sx0:sx1]
            alpha = 1.0 - (self.disintegration_counter / max(1, DISINTEGRATE_FRAMES))
            frame = cv2.addWeighted(dust_frame, 1.0 - alpha, frame, alpha, 0)
            self.disintegration_counter += 1
            if self.disintegration_counter > DISINTEGRATE_FRAMES:
                self.disintegration_active = False
                self.white_after_disintegration = True
                self.white_show_start = time.time()
            return frame

        # white after disintegration handling (unchanged)
        if self.white_after_disintegration:
            t = now - self.white_show_start
            if t < 0.6:
                alpha = t / 0.6
                white = np.ones_like(frame, dtype=np.uint8) * 255
                frame = cv2.addWeighted(white, alpha, frame, 1.0 - alpha, 0)
                return frame
            else:
                return np.ones_like(frame, dtype=np.uint8) * 255

        # get left hand landmarks if not provided and mediapipe available
        left_landmarks = None
        left_label = None
        if hand_landmarks is None and self.hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            if res and res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    if label == "Left":
                        left_landmarks = lm
                        left_label = label
                        break
        elif hand_landmarks is not None:
            # accept single list or first element
            try:
                left_landmarks = hand_landmarks[0]
            except Exception:
                left_landmarks = hand_landmarks
                left_label = handedness_label

        snap_happened = False
        back_center = None

        if left_landmarks is not None:
            orientation = get_orientation(left_landmarks, left_label)
            back_center = compute_back_of_hand_center(left_landmarks, w, h)
            for s in self.stones:
                s.update(left_landmarks, w, h, back_center)
            is_back = (orientation == "Back")
            fist = is_fist_by_landmarks(left_landmarks, w, h)

            thumb_tip = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], w, h)
            middle_tip = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], w, h)
            idx_mcp = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], w, h)
            pky_mcp = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], w, h)
            palm_w = max(1.0, dist(idx_mcp, pky_mcp))
            cur_dist = dist(thumb_tip, middle_tip)
            touch_thr = TOUCH_FRAC * palm_w
            release_thr = RELEASE_FRAC * palm_w

            if self.prev_time is None:
                dt = None
            else:
                dt = max(1e-6, now - self.prev_time)
            sep_speed = 0.0
            if self.prev_dist is not None and dt is not None:
                sep_speed = (cur_dist - self.prev_dist) / dt
            vel_thr = VEL_THRESH_FRAC * palm_w

            if cur_dist < touch_thr:
                if not self.armed:
                    self.armed = True
                    self.armed_since = now
            else:
                if self.armed and (now - self.armed_since) < MIN_HOLD_TIME:
                    self.armed = False

            # only allow snap if hand glow active
            if self.armed and (cur_dist > release_thr) and (sep_speed > vel_thr) and self.hand_glow_active:
                if (now - self.last_snap_time) > COOLDOWN_SEC:
                    snap_happened = True
                    self.last_snap_time = now
                self.armed = False

            self.prev_dist = cur_dist
            self.prev_time = now

            # fist hold -> spawn stones and enable hand glow
            if is_back and fist:
                if self.fist_start_time is None:
                    self.fist_start_time = now
                else:
                    held = now - self.fist_start_time
                    if (not self.stones_present) and held >= FIST_HOLD_SECONDS:
                        self.stones_present = True
                        for s in self.stones:
                            s.state = "fading_in"
                            s.spawn_t = now
                        # enable hand glow when stones appear
                        self.hand_glow_active = True
                        self.hand_glow_start = now
                        self.snap_allowed = True
                        self.ending = True
            else:
                if self.fist_start_time is not None:
                    self.fist_start_time = None
                if self.stones_present:
                    for s in self.stones:
                        if s.state in ("fading_in", "active"):
                            s.state = "fading_out"
                            s.spawn_t = now
                    self.stones_present = False
        else:
            # no left hand: start fading stones out if present and disable armed
            self.fist_start_time = None
            self.prev_dist = None
            self.prev_time = None
            self.armed = False
            if self.stones_present:
                for s in self.stones:
                    if s.state in ("fading_in", "active"):
                        s.state = "fading_out"
                        s.spawn_t = now
                self.stones_present = False
                

        if snap_happened:
            self.disintegration_active = True
            self.disintegration_counter = 0
            self.disintegration_particles = None
            for s in self.stones:
                if s.state in ("fading_in", "active"):
                    s.state = "fading_out"
                    s.spawn_t = now
            self.yellow_flash_active = True
            self.yellow_flash_start = now

        # update stone alpha states (unchanged)
        for s in self.stones:
            prev_state = s.state
            if s.state == "fading_in":
                dt = now - s.spawn_t
                s.alpha = np.clip(dt / max(1e-6, STONE_FADE_IN), 0.0, 1.0)
                if dt >= STONE_FADE_IN:
                    s.state = "active"
                    s.alpha = 1.0
            elif s.state == "active":
                s.alpha = 1.0
            elif s.state == "fading_out":
                dt = now - s.spawn_t
                s.alpha = np.clip(1.0 - (dt / max(1e-6, STONE_FADE_OUT)), 0.0, 1.0)
                if s.alpha <= 0.001:
                    s.state = "off"
                    s.alpha = 0.0
            else:
                s.alpha = 0.0
            if prev_state != "fading_out" and s.state == "fading_out":
                self.yellow_flash_active = True
                self.yellow_flash_start = now

        # if the hand glow is active, check duration and expire after HAND_GLOW_DURATION
        if self.hand_glow_active:
            if now - self.hand_glow_start >= HAND_GLOW_DURATION:
                self.hand_glow_active = False
                self.snap_allowed = False

        out = frame.copy()
        any_visible = any(s.state != "off" and s.alpha > 0.01 for s in self.stones)
        palm_w_est = 70
        if left_landmarks is not None:
            idx_mcp = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], w, h)
            pky_mcp = lm_xy(left_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], w, h)
            palm_w_est = max(28, int(dist(idx_mcp, pky_mcp)))

        # render stones and glows
        if any_visible:
            glow = np.zeros_like(out, dtype=np.float32)
            for s in self.stones:
                if s.state == "off" or s.alpha <= 0.01:
                    continue
                px, py = int(s.pos[0]), int(s.pos[1])
                r = int(max(8, palm_w_est * 0.14))
                for g in range(STONE_GLOW_STEPS, 0, -1):
                    gr = int(r * (1.0 + 0.18 * g))
                    ga = s.alpha * (0.08 * (STONE_GLOW_STEPS - g + 1))
                    color = np.array(s.color, dtype=np.float32) / 255.0
                    cv2.circle(glow, (px, py), gr, tuple((color * ga).tolist()), -1, cv2.LINE_AA)
            try:
                glow_uint = np.clip(glow * 150.0, 0, 255).astype(np.uint8)
                glow_uint = cv2.GaussianBlur(glow_uint, (7,7), 0)
                out = cv2.addWeighted(out, 1.0, glow_uint, 0.8, 0)
            except Exception:
                pass
            for s in self.stones:
                if s.state == "off" or s.alpha <= 0.01:
                    continue
                px, py = int(s.pos[0]), int(s.pos[1])
                r = int(max(8, palm_w_est * 0.14))
                rx = int(r * 1.1); ry = int(r * 0.85)
                out = draw_simple_gradient_ellipse(out, (px, py), rx, ry, s.color, s.alpha)

        # render hand glow (if active) so it visually ties to the stones
        if self.hand_glow_active and left_landmarks is not None:
            out = self._render_hand_glow(out, left_landmarks, palm_w_est, now)

        # yellow fullscreen radiating flash (unchanged)
        if self.yellow_flash_active:
            t = now - self.yellow_flash_start
            if t <= YELLOW_FLASH_DURATION:
                frac = t / YELLOW_FLASH_DURATION
                alpha = np.clip(1.0 - (1.0 - frac)**2, 0.0, 1.0) * 0.9
                if back_center is None:
                    center_xy = np.array([w//2, h//2])
                else:
                    center_xy = back_center.astype(int)
                maxr = int(math.hypot(w, h))
                radius = int(frac * maxr)
                overlay = np.zeros_like(out, dtype=np.uint8)
                step = max(1, int(max(12, radius//20)))
                for rr in range(radius, 0, -step):
                    cv2.circle(overlay, tuple(center_xy.tolist()), rr, (0, 220, 255), -1)
                out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)
            else:
                self.yellow_flash_active = False

        return out

    def close(self):
        if self.hands is not None:
            self.hands.close()

# --- standalone demo wrapper (optional) ---
def gauntlet():
    rnd = InfinityGauntletRenderer(use_mediapipe=True)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = rnd.process_frame(frame)
            cv2.imshow("Infinity Gauntlet", out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        rnd.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    gauntlet()
