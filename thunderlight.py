"""Glowing lightning renderer (no overlays).
One-line parts: helpers, drawing, LightningRenderer class, standalone lightning() wrapper.
"""

import cv2
import numpy as np
import time
import math
import random

# optional mediapipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception:
    mp_hands = None
    mp_drawing = None
    USE_MEDIAPIPE = False

# ---------- Parameters (defaults preserved) ----------
MAX_HANDS = 2
DETECT_CONF = 0.7
TRACK_CONF = 0.7
MAX_DIST_PIX = 380
LIGHT_INTENSITY = 1
GLOW_BLUR = 41
CORE_BLUR = 9
DOWNSCALE = 3

COLOR_PRESETS = [
    ("Blue",   (2.55, 1.28, 0)),
    ("Red",    (0, 1.28, 2.55)),
    ("Purple", (2.55, 0, 2.55)),
]

CORE_COLOR = (2.0, 2.0, 2.0)
NUM_SEGMENTS = 14
JITTER_PIX = 15
BRANCH_PROB = 0.35
BRANCH_LENGTH_FACTOR = 0.28
BRANCH_JITTER = 12
THICKNESS_SCALE = 2.0
COLOR_INTENSITY = 0.35
FLICKER_FREQ = 8.0
FLICKER_AMP = 0.18
EXTRA_STROKES = 4

# ----------------------------------------

def ensure_odd(k):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return max(1, k)


def palm_center(hand_landmarks, w, h):
    ids = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP,
    ]
    x = np.mean([hand_landmarks.landmark[i].x * w for i in ids])
    y = np.mean([hand_landmarks.landmark[i].y * h for i in ids])
    return int(x), int(y)


def hand_center_3d(hand_landmarks, w, h):
    ids = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP,
    ]
    xs = [hand_landmarks.landmark[i].x * w for i in ids]
    ys = [hand_landmarks.landmark[i].y * h for i in ids]
    zs = [hand_landmarks.landmark[i].z * w for i in ids]
    return np.array([np.mean(xs), np.mean(ys), np.mean(zs)], dtype=np.float32)


def palm_normal_3d(hand_landmarks, w, h):
    try:
        wr = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        im = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pm = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    wr_v = np.array([wr.x * w, wr.y * h, wr.z * w], dtype=np.float32)
    im_v = np.array([im.x * w, im.y * h, im.z * w], dtype=np.float32)
    pm_v = np.array([pm.x * w, pm.y * h, pm.z * w], dtype=np.float32)
    v1 = im_v - wr_v
    v2 = pm_v - wr_v
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n / norm


def is_hand_open(hand_landmarks, w, h):
    try:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    except Exception:
        return False
    wrist_xy = np.array([wrist.x * w, wrist.y * h], dtype=np.float32)
    tip_ids = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    tips = []
    for i in tip_ids:
        lm = hand_landmarks.landmark[i]
        tips.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    dists = [np.linalg.norm(t - wrist_xy) for t in tips]
    avg_tip_dist = float(np.mean(dists))
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    palm_size = float(np.linalg.norm(np.array([middle_mcp.x * w, middle_mcp.y * h]) - wrist_xy))
    if palm_size < 1e-6:
        return False
    return avg_tip_dist > (1.45 * palm_size)


def generate_lightning_points(p1, p2, num_segments=NUM_SEGMENTS, jitter=JITTER_PIX):
    points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        x = int(p1[0] * (1 - t) + p2[0] * t)
        y = int(p1[1] * (1 - t) + p2[1] * t)
        if 0 < i < num_segments:
            x += random.randint(-jitter, jitter)
            y += random.randint(-jitter, jitter)
        points.append((x, y))
    return points


def generate_branches(points, p1, p2):
    branches = []
    n = len(points)
    for i in range(1, n - 1):
        if random.random() < BRANCH_PROB:
            origin = points[i]
            dx = points[i + 1][0] - points[i - 1][0]
            dy = points[i + 1][1] - points[i - 1][1]
            bx = -dy
            by = dx
            normv = max(1e-6, (abs(bx) + abs(by)))
            bx = int(bx / normv)
            by = int(by / normv)
            main_len = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            blen = max(8, int(main_len * BRANCH_LENGTH_FACTOR * random.uniform(0.6, 1.2)))
            ex = origin[0] + bx * blen + random.randint(-BRANCH_JITTER, BRANCH_JITTER)
            ey = origin[1] + by * blen + random.randint(-BRANCH_JITTER, BRANCH_JITTER)
            branches.append([origin, (ex, ey)])
    return branches


def draw_lightning_layer(h, w, p1, p2, flicker=1.0, preset_color=(0.0, 0.0, 1.0), color_intensity=0.35,
                         thickness_scale=1.0):
    sw, sh = max(1, w // DOWNSCALE), max(1, h // DOWNSCALE)
    scale_x = w / sw
    scale_y = h / sh
    p1s = (int(p1[0] / scale_x), int(p1[1] / scale_y))
    p2s = (int(p2[0] / scale_x), int(p2[1] / scale_y))
    canvas = np.zeros((sh, sw, 3), dtype=np.float32)
    jitter = max(1, int(max(1, JITTER_PIX) / max(1, DOWNSCALE)))
    pts = generate_lightning_points((p1s[0], p1s[1]), (p2s[0], p2s[1]), num_segments=NUM_SEGMENTS, jitter=jitter)
    pts_np = np.array(pts, np.int32)
    temp = np.zeros((sh, sw, 3), dtype=np.uint8)
    core_thk = max(1, int(max(3, 6 * thickness_scale) // max(1, DOWNSCALE)))
    cv2.polylines(temp, [pts_np], False, (255, 255, 255), thickness=core_thk, lineType=cv2.LINE_AA)
    k_core = ensure_odd(max(3, CORE_BLUR // max(1, DOWNSCALE)))
    temp = cv2.GaussianBlur(temp, (k_core, k_core), 0)
    canvas += (temp.astype(np.float32) / 255.0) * (1.0 * flicker * 1.6)
    edge_bgr = (int(preset_color[0] * 255), int(preset_color[1] * 255), int(preset_color[2] * 255))
    stroke_specs = [
        (int(max(1, 5 * thickness_scale)), 0.9),
        (int(max(1, 3 * thickness_scale)), 0.6),
        (int(max(1, 2 * thickness_scale)), 0.4),
    ]
    for extra in range(EXTRA_STROKES):
        thickness = int(max(1, random.choice([2, 3, 4, 5]) * thickness_scale))
        strength = random.uniform(0.18, 0.45)
        stroke_specs.append((thickness, strength))
    for thickness, strength in stroke_specs:
        temp2 = np.zeros((sh, sw, 3), dtype=np.uint8)
        th = max(1, int(thickness // max(1, DOWNSCALE)))
        cv2.polylines(temp2, [pts_np], False, edge_bgr, thickness=th, lineType=cv2.LINE_AA)
        branches = generate_branches(pts, p1s, p2s)
        for br in branches:
            br_pts = np.array(br, np.int32)
            cv2.polylines(temp2, [br_pts], False, edge_bgr, thickness=max(1, int((thickness - 1) // max(1, DOWNSCALE))), lineType=cv2.LINE_AA)
        k = ensure_odd(max(3, GLOW_BLUR // max(1, DOWNSCALE)))
        temp2 = cv2.GaussianBlur(temp2, (k, k), 0)
        canvas += (temp2.astype(np.float32) / 255.0) * (flicker * strength)
    temp3 = np.zeros((sh, sw, 3), dtype=np.uint8)
    for _ in range(2):
        jittered = []
        for (x, y) in pts:
            jittered.append((int(x + random.randint(-3, 3)), int(y + random.randint(-3, 3))))
        jnp = np.array(jittered, np.int32)
        cv2.polylines(temp3, [jnp], False, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    temp3 = cv2.GaussianBlur(temp3, (5, 5), 0)
    canvas += (temp3.astype(np.float32) / 255.0) * (0.95 * flicker)
    kfinal = ensure_odd(max(3, GLOW_BLUR // max(1, DOWNSCALE)))
    canvas = cv2.GaussianBlur((np.clip(canvas, 0.0, 6.0) * 255.0).astype(np.uint8), (kfinal, kfinal), 0).astype(np.float32) / 255.0
    canvas_up = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    color_tint = np.stack([np.full((h, w), preset_color[0], dtype=np.float32), np.full((h, w), preset_color[1], dtype=np.float32), np.full((h, w), preset_color[2], dtype=np.float32)], axis=2)
    luminance = np.mean(canvas_up, axis=2, keepdims=True)
    mix_white = np.clip(1.5 * luminance * (1.0 - color_intensity), 0.0, 1.0)
    light_map = canvas_up * (np.array(CORE_COLOR, dtype=np.float32).reshape(1, 1, 3)) * mix_white + canvas_up * color_tint * (1.0 - mix_white)
    light_map = np.clip(light_map, 0.0, 3.0)
    return light_map


def composite_soft_core(frame, p1, p2, thickness_scale=1.0):
    h, w = frame.shape[:2]
    temp = np.zeros((h, w, 3), dtype=np.uint8)
    pts = generate_lightning_points(p1, p2, num_segments=NUM_SEGMENTS, jitter=max(4, JITTER_PIX // 6))
    pts_np = np.array(pts, np.int32)
    core_thk = max(1, int(4 * thickness_scale))
    cv2.polylines(temp, [pts_np], False, (255, 255, 255), thickness=core_thk, lineType=cv2.LINE_AA)
    k = ensure_odd(max(5, CORE_BLUR * 2))
    blurred = cv2.GaussianBlur(temp, (k, k), 0)
    frame_float = frame.astype(np.float32) / 255.0
    glow_float = blurred.astype(np.float32) / 255.0
    out = np.clip(frame_float + glow_float * 0.9, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def apply_light_map_multiply(frame, light_map, intensity=LIGHT_INTENSITY):
    img_f = frame.astype(np.float32) / 255.0
    add = img_f * (light_map * intensity)
    out = img_f + add
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ---------------- Renderer class ----------------
class ThunderLightningRenderer:
    """Stateful lightning renderer. Call process_frame(frame, hand_landmarks_list=None).
    Detects two open palms and renders glowing lightning between them. No overlays drawn by default."""

    def __init__(self, use_mediapipe=USE_MEDIAPIPE, preset_index=0):
        self.use_mediapipe = use_mediapipe and (mp_hands is not None)
        self._local_hands = None
        if self.use_mediapipe:
            try:
                self._local_hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=DETECT_CONF, min_tracking_confidence=TRACK_CONF)
            except Exception:
                self._local_hands = None
        self.preset_index = preset_index
        self.jitter = JITTER_PIX
        self.thickness_scale = THICKNESS_SCALE
        self.color_intensity = COLOR_INTENSITY
        self.light_intensity = LIGHT_INTENSITY
        # disable overlays by default
        self.debug = False
        # timing
        self.t0 = time.time()

    def process_frame(self, frame, hand_landmarks_list=None, handedness_list=None, now=None):
        if now is None:
            now = time.time()
        h, w = frame.shape[:2]
        palms = []
        hands_info = []
        hands_list = None
        # prefer provided landmarks
        if hand_landmarks_list is not None:
            hands_list = hand_landmarks_list
        elif self._local_hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._local_hands.process(rgb)
            hands_list = res.multi_hand_landmarks if res and res.multi_hand_landmarks else None
        if hands_list:
            for hand_landmarks in hands_list:
                try:
                    cx, cy = palm_center(hand_landmarks, w, h)
                except Exception:
                    continue
                palms.append((cx, cy))
                center3 = hand_center_3d(hand_landmarks, w, h)
                normal3 = palm_normal_3d(hand_landmarks, w, h)
                open_flag = is_hand_open(hand_landmarks, w, h)
                hands_info.append({'center2d': (cx, cy), 'center3d': center3, 'normal3': normal3, 'is_open': open_flag, 'lm': hand_landmarks})
        # lightning trigger
        if len(hands_info) == 2:
            p1 = hands_info[0]['center2d']
            p2 = hands_info[1]['center2d']
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            both_open = hands_info[0]['is_open'] and hands_info[1]['is_open']
            c1 = hands_info[0]['center3d']; c2 = hands_info[1]['center3d']
            dir12 = c2 - c1; n12 = np.linalg.norm(dir12)
            facing = False
            if n12 >= 1e-6:
                dir12_u = dir12 / n12; dir21_u = -dir12_u
                n1 = hands_info[0]['normal3'].astype(np.float32); n2 = hands_info[1]['normal3'].astype(np.float32)
                raw_dot1 = float(np.dot(n1, dir12_u)); raw_dot2 = float(np.dot(n2, dir21_u))
                if raw_dot1 < 0.0: n1 = -n1
                if raw_dot2 < 0.0: n2 = -n2
                normal_dot = float(np.dot(n1, n2))
                facing = (normal_dot > -0.5) and (normal_dot < 0.5)
            if dist < MAX_DIST_PIX and both_open and facing:
                flick = 1.0 + (np.sin((now - self.t0) * FLICKER_FREQ * 2.0 * np.pi) * FLICKER_AMP) + (random.random() - 0.5) * 0.08
                flick = float(np.clip(flick, 0.5, 1.6))
                preset_name, preset_col = COLOR_PRESETS[self.preset_index]
                light_map = draw_lightning_layer(h, w, p1, p2, flicker=flick, preset_color=preset_col, color_intensity=self.color_intensity, thickness_scale=self.thickness_scale)
                frame = apply_light_map_multiply(frame, light_map, intensity=self.light_intensity)
                frame = composite_soft_core(frame, p1, p2, thickness_scale=self.thickness_scale)
        return frame


# standalone demo
def lightning():
    renderer = ThunderLightningRenderer()
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            out = renderer.process_frame(frame)
            cv2.imshow("Glowing Lightning", out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        cap.release()
        if renderer._local_hands is not None:
            renderer._local_hands.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    lightning()
