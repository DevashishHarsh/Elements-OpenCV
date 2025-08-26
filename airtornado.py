import cv2
import os
import time
import math
import random
import numpy as np
import mediapipe as mp

# TornadoOverlay: importable class that overlays tornado sprite when two index fingers meet
# - place sprite frames in a folder "tornado/0001.png".. etc
# - process_frame(frame, hand_landmarks_list=None, handedness_list=None) -> returns annotated frame

class AirTornadoRenderer:
    def __init__(self, frame_dir='tornado', total_frames=100,
                 frame_w=640, frame_h=480,
                 use_mediapipe=True,
                 formation_time=0.9):
        self.SPRITE_FOLDER = frame_dir
        self.TOTAL_FRAMES = total_frames
        self.FRAME_W = frame_w
        self.FRAME_H = frame_h
        self.FORMATION_TIME = formation_time
        self.use_mediapipe = use_mediapipe

        # visual tuning (defaults from original)
        self.size_multiplier = 0.6
        self.anim_play_speed = 1.0
        self.swirl_speed = 0.10
        self.particle_rate = 5
        self.max_particles = 400
        self.blur_strength = 2
        self.ANIM_FPS = 12
        self.SPRITE_BLUR_SIGMA = 0.2
        self.SPRITE_ALPHA_BLUR_SIGMA = 0.2
        self.sprite_saturation = 0.2
        self.MAX_DESAT_ALPHA = 0.7
        self.MAX_TINT_ALPHA = 0.5
        self.TINT_COLOR = np.array([80,90,100], dtype=np.uint8)

        # load frames
        self.frames = [None] * (self.TOTAL_FRAMES + 1)
        self._load_frames()

        # mediapipe (only create if use_mediapipe is True and no external landmarks will be provided)
        self.mp_hands = mp.solutions.hands
        self._local_hands = None
        if self.use_mediapipe:
            try:
                self._local_hands = self.mp_hands.Hands(
                    static_image_mode=False, 
                    max_num_hands=2,
                    min_detection_confidence=0.6, 
                    min_tracking_confidence=0.5
                )
            except Exception:
                self._local_hands = None

        # particle state
        self.particles = []
        self.rotation_phase = 0.0
        self.formation = 0.0
        self.active = False
        self.anim_index = 30.0
        self.anim_time = 0.0  # Time accumulator for animation
        self.prev_time = time.time()
        self.debug_on = False

    def _load_frames(self):
        for i in range(1, self.TOTAL_FRAMES + 1):
            fname = os.path.join(self.SPRITE_FOLDER, f"{i:04d}.png")
            if os.path.isfile(fname):
                img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                self.frames[i] = img

    # helpers
    def _norm(self, v):
        m = math.hypot(v[0], v[1])
        return (v[0]/m, v[1]/m) if m > 1e-6 else (0.0, 0.0)

    def _dot(self, a, b):
        return a[0]*b[0] + a[1]*b[1]

    def _line_intersection(self, p1, v1, p2, v2):
        A = np.array([[v1[0], -v2[0]],[v1[1], -v2[1]]], dtype=np.float32)
        b = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-5:
            return None
        t, s = np.linalg.solve(A, b)
        return (p1[0] + t*v1[0], p1[1] + t*v1[1])

    def _spawn_particle(self, center, formation_p):
        ang = random.random()*2*math.pi
        r = random.uniform(4, 40) * (0.25 + 0.75*formation_p) * self.size_multiplier
        vy = random.uniform(0.6, 1.8) * (0.4 + 0.6*formation_p)
        life = random.uniform(0.6, 1.6)
        self.particles.append({'ang': ang, 'r': r, 'y': center[1] + random.uniform(-6,6),
                               'vy': vy, 'age':0.0, 'life':life, 'sw': random.uniform(-0.2,0.2)})

    def _update_particles(self, dt, center):
        keep = []
        for p in self.particles:
            p['ang'] += (self.swirl_speed*0.7 + p['sw']) * dt * 60.0
            p['y'] -= p['vy'] * dt * 60.0
            p['r'] *= 0.997
            p['age'] += dt
            if p['age'] < p['life'] and 0 <= p['y'] < self.FRAME_H and p['r'] > 0.5:
                keep.append(p)
        self.particles[:] = keep

    # overlay helper (from original) - draws fg (with alpha) centered at x,y scaled
    def _overlay_png_alpha(self, bg, fg, x, y, scale=1.0, blur_bg=True, sprite_blur=True, sprite_alpha_blur=True, sprite_sat=1.0):
        h_bg, w_bg = bg.shape[:2]
        fh = int(fg.shape[0] * scale)
        fw = int(fg.shape[1] * scale)
        if fh <= 0 or fw <= 0:
            return bg
        fg_resized = cv2.resize(fg, (fw, fh), interpolation=cv2.INTER_AREA)
        if fg_resized.shape[2] == 4:
            fg_rgb = fg_resized[:, :, :3].copy()
            fg_a = fg_resized[:, :, 3].copy()
        else:
            fg_rgb = fg_resized[:, :, :3].copy()
            fg_a = np.full((fh, fw), 255, dtype=np.uint8)
        # saturation simple
        if abs(sprite_sat - 1.0) > 1e-3:
            hsv = cv2.cvtColor(fg_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] = np.clip(hsv[...,1] * sprite_sat, 0, 255)
            fg_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # blur sprite rgb/alpha
        if sprite_blur and self.SPRITE_BLUR_SIGMA > 0:
            k = int(max(3, round(self.SPRITE_BLUR_SIGMA*3)*2+1))
            fg_rgb = cv2.GaussianBlur(fg_rgb, (k,k), sigmaX=self.SPRITE_BLUR_SIGMA)
        if sprite_alpha_blur and self.SPRITE_ALPHA_BLUR_SIGMA > 0:
            k = int(max(3, round(self.SPRITE_ALPHA_BLUR_SIGMA*3)*2+1))
            fg_a = cv2.GaussianBlur(fg_a, (k,k), sigmaX=self.SPRITE_ALPHA_BLUR_SIGMA)
        x1 = int(x - fw//2); y1 = int(y - fh//2)
        x2 = x1 + fw; y2 = y1 + fh
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w_bg, x2), min(h_bg, y2)
        if bx1 >= bx2 or by1 >= by2:
            return bg
        sx1 = bx1 - x1; sy1 = by1 - y1
        sx2 = sx1 + (bx2 - bx1); sy2 = sy1 + (by2 - by1)
        fg_crop_rgb = fg_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
        fg_crop_a = fg_a[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
        bg_roi = bg[by1:by2, bx1:bx2].astype(np.float32)
        if blur_bg and self.blur_strength > 0:
            k = int(max(3, round(self.blur_strength*3)*2+1))
            blurred = cv2.GaussianBlur(bg_roi.astype(np.uint8), (k,k), sigmaX=self.blur_strength)
            bg_roi = cv2.addWeighted(bg_roi, 0.7, blurred.astype(np.float32), 0.3, 0)
        alpha_3 = np.dstack([fg_crop_a, fg_crop_a, fg_crop_a])
        comp = alpha_3 * fg_crop_rgb + (1 - alpha_3) * bg_roi
        bg[by1:by2, bx1:bx2] = np.clip(comp, 0, 255).astype(np.uint8)
        return bg

    # Updated public API: pass BGR frame and optional hand landmarks
    def process_frame(self, frame_bgr, hand_landmarks_list=None, handedness_list=None):
        if frame_bgr is None:
            return None
        frame = cv2.resize(frame_bgr, (self.FRAME_W, self.FRAME_H))
        out = frame.copy()
        now = time.time()
        dt = now - self.prev_time if self.prev_time is not None else 0.016
        self.prev_time = now

        tips = []
        dirs = []
        tornado_center = None
        form_ok = False

        # Use provided hand landmarks if available, otherwise use internal mediapipe
        hands_list = None
        if hand_landmarks_list is not None:
            hands_list = hand_landmarks_list
        elif self._local_hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._local_hands.process(rgb)
            hands_list = res.multi_hand_landmarks if res and res.multi_hand_landmarks else None

        if hands_list:
            handedness = []
            # Use provided handedness if available
            if handedness_list is not None:
                handedness = handedness_list
            # If we used internal mediapipe, try to get handedness from result
            elif self._local_hands is not None and hasattr(res, 'multi_handedness') and res.multi_handedness:
                for hd in res.multi_handedness:
                    handedness.append(hd.classification[0].label)
            
            # Process up to 2 hands for tornado detection
            for i, lmks in enumerate(hands_list[:2]):
                try:
                    tip = (lmks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.FRAME_W, 
                           lmks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.FRAME_H)
                    pip = (lmks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x * self.FRAME_W, 
                           lmks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y * self.FRAME_H)
                    d = (tip[0] - pip[0], tip[1] - pip[1])
                    dn = self._norm(d)
                    tips.append(tip)
                    dirs.append(dn)
                except Exception:
                    continue

        # Tornado formation logic (requires exactly 2 hands)
        if len(tips) == 2:
            p1, p2 = tips[0], tips[1]
            v1, v2 = dirs[0], dirs[1]
            v_p12 = self._norm((p2[0]-p1[0], p2[1]-p1[1]))
            v_p21 = self._norm((p1[0]-p2[0], p1[1]-p2[1]))
            dot1 = self._dot(v1, v_p12); dot2 = self._dot(v2, v_p21)
            upward_ok = (v1[1] < -0.07) and (v2[1] < -0.07)
            if dot1 > 0.42 and dot2 > 0.42 and upward_ok:
                inter = self._line_intersection(p1, v1, p2, v2)
                if inter is not None:
                    ix, iy = inter
                    if -self.FRAME_W*0.08 < ix < self.FRAME_W*1.08 and -self.FRAME_H*0.08 < iy < self.FRAME_H*1.08:
                        tornado_center = inter
                        form_ok = True

        # formation logic
        if form_ok:
            self.formation = min(1.0, self.formation + dt / self.FORMATION_TIME)
            self.active = True
        else:
            self.formation = max(0.0, self.formation - dt / (self.FORMATION_TIME * 0.6))
            if self.formation == 0.0:
                self.active = False

        # update rotation
        self.rotation_phase += self.swirl_speed * dt * 60.0

        # spawn particles
        if self.active and tornado_center is not None:
            for _ in range(int(self.particle_rate * (0.4 + 0.6*self.formation))):
                if len(self.particles) < self.max_particles:
                    self._spawn_particle(tornado_center, self.formation)
        self._update_particles(dt, tornado_center if tornado_center else (self.FRAME_W//2, self.FRAME_H//2))

        # apply tint/desat when forming
        if self.formation > 0.001:
            desat_alpha = self.MAX_DESAT_ALPHA * self.formation
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out.astype(np.float32), 1.0 - desat_alpha, gray_bgr.astype(np.float32), desat_alpha, 0).astype(np.uint8)
            tint_alpha = self.MAX_TINT_ALPHA * self.formation
            tint_layer = np.full_like(out, self.TINT_COLOR)
            out = cv2.addWeighted(out.astype(np.float32), 1.0 - tint_alpha, tint_layer.astype(np.float32), tint_alpha, 0).astype(np.uint8)

        # overlay sprite if formed
        if tornado_center is not None and self.formation > 0.01:
            cx, cy = int(tornado_center[0]), int(tornado_center[1])
            # choose frame index based on ANIM_FPS
            if self.formation < 0.999:
                # Formation phase: interpolate from frame 1 to 30
                fidx = int(1 + (30 - 1) * self.formation + 0.5)
            elif self.active:
                # Active phase: animate from frame 30 to 70 at ANIM_FPS rate
                self.anim_time += dt
                frames_per_second = self.ANIM_FPS * self.anim_play_speed
                frame_duration = 1.0 / frames_per_second if frames_per_second > 0 else 1.0/30.0
                
                # Calculate how many animation frames to advance
                frames_to_advance = self.anim_time / frame_duration
                self.anim_index += frames_to_advance
                self.anim_time = 0.0  # Reset time accumulator
                
                # Loop animation between frames 30-70
                L = 70 - 30 + 1  # Length of animation loop (41 frames)
                rel = (self.anim_index - 30) % L
                fidx = int(30 + rel)
            else:
                # Dissipation phase: interpolate from frame 70 to 100
                fall = 1.0 - self.formation
                fidx = int(70 + (100 - 70) * fall + 0.5)
            
            fidx = max(1, min(self.TOTAL_FRAMES, fidx))
            sprite = self.frames[fidx]
            if sprite is not None:
                grow = 0.35 + 0.65 * self.formation
                scale = self.size_multiplier * grow
                out = self._overlay_png_alpha(out, sprite, cx, cy, scale=scale,
                                               blur_bg=True, sprite_blur=True, sprite_alpha_blur=True, sprite_sat=self.sprite_saturation)

        # optional debug drawing for tips/center
        if self.debug_on:
            for t in tips:
                cv2.circle(out, (int(t[0]), int(t[1])), 5, (80,200,255), -1)
            if tornado_center is not None:
                cv2.circle(out, (int(tornado_center[0]), int(tornado_center[1])), 6, (255,200,80), -1)

        # resize back to original input if needed
        out = cv2.resize(out, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return out

    def close(self):
        """Clean up resources."""
        if self._local_hands is not None:
            self._local_hands.close()

# Demo usage when run directly
if __name__ == '__main__':
    tog = AirTornadoRenderer(frame_dir='tornado', total_frames=100, frame_w=640, frame_h=480, use_mediapipe=True)
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = tog.process_frame(frame)
            cv2.imshow('Tornado Overlay', out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        tog.close()
        cv2.destroyAllWindows()