import cv2
import mediapipe as mp
import numpy as np
import math
from fireball import FireFlameRenderer
from airtornado import AirTornadoRenderer
from gauntlet import InfinityGauntletRenderer
from thunderlight import ThunderLightningRenderer
from watershield import WaterShieldRenderer

# config
W, H = 640, 480
CAP_IDX = 0
FIRE_DIR = "flames"
TORNADO_DIR = "tornado"
FRAME_EXT = "png"
end = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# keep your orientation function exactly
def get_orientation(hand_landmarks, handedness):
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z])
    index_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z])
    pinky_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z])

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)

    # Flip for Right hand to account for cross-product direction
    if handedness == "Right":
        normal = -normal

    return "Front" if normal[2] < 0 else "Back"

# helper: 3D point in consistent units (scale z by width)
def landmark_xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)

# compute finger open/closed: uses distance-to-wrist heuristic (robust to rotation)
def compute_finger_states(hand_landmarks, w, h):
    # measure palm width (index_mcp <-> pinky_mcp)
    try:
        idx_mcp = landmark_xyz(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], w, h)
        pky_mcp = landmark_xyz(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP], w, h)
    except Exception:
        return [0,0,0,0,0]

    palm_w = max(1e-6, np.linalg.norm(idx_mcp - pky_mcp))

    # wrist reference
    wrist = landmark_xyz(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST], w, h)

    def dist_to_wrist(idx):
        lm = landmark_xyz(hand_landmarks.landmark[idx], w, h)
        return float(np.linalg.norm(lm - wrist))

    # thresholds scaled by palm size (use slightly >1 multiplier)
    # small fingers slightly smaller multiplier since tip/pip distances differ
    thumb_thresh = 1.03
    finger_thresh = 1.02

    # thumb: tip (4) vs ip (3)
    d_thumb_tip = dist_to_wrist(mp_hands.HandLandmark.THUMB_TIP)
    d_thumb_ip  = dist_to_wrist(mp_hands.HandLandmark.THUMB_IP)
    thumb_open = 1 if d_thumb_tip > d_thumb_ip * thumb_thresh else 0

    # index: tip (8) vs pip (6)
    d_index_tip = dist_to_wrist(mp_hands.HandLandmark.INDEX_FINGER_TIP)
    d_index_pip = dist_to_wrist(mp_hands.HandLandmark.INDEX_FINGER_PIP)
    index_open = 1 if d_index_tip > d_index_pip * finger_thresh else 0

    # middle: tip (12) vs pip (10)
    d_middle_tip = dist_to_wrist(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
    d_middle_pip = dist_to_wrist(mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    middle_open = 1 if d_middle_tip > d_middle_pip * finger_thresh else 0

    # ring: tip (16) vs pip (14)
    d_ring_tip = dist_to_wrist(mp_hands.HandLandmark.RING_FINGER_TIP)
    d_ring_pip = dist_to_wrist(mp_hands.HandLandmark.RING_FINGER_PIP)
    ring_open = 1 if d_ring_tip > d_ring_pip * finger_thresh else 0

    # pinky: tip (20) vs pip (18)
    d_pinky_tip = dist_to_wrist(mp_hands.HandLandmark.PINKY_TIP)
    d_pinky_pip = dist_to_wrist(mp_hands.HandLandmark.PINKY_PIP)
    pinky_open = 1 if d_pinky_tip > d_pinky_pip * finger_thresh else 0

    return [thumb_open, index_open, middle_open, ring_open, pinky_open]

# small drawing helper
def draw_text(img, text, x, y, color=(0,0,0)):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def main():
    global end
    DEBUG = True
    cap = cv2.VideoCapture(CAP_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    flame = FireFlameRenderer(frame_dir=FIRE_DIR, frame_ext=FRAME_EXT, num_frames=100, use_mediapipe=False)
    water = WaterShieldRenderer(use_mediapipe=False)
    tornado = AirTornadoRenderer(frame_dir=TORNADO_DIR, total_frames=100, use_mediapipe=False, frame_w=W, frame_h=H)
    thunder = ThunderLightningRenderer(use_mediapipe=False)
    gauntlet = InfinityGauntletRenderer(use_mediapipe=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # defaults
        finger_state_left  = [0,0,0,0,0]
        finger_state_right = [0,0,0,0,0]
        left_hand_front = 0
        right_hand_front = 0
        label=None
        right_is = False
        left_is = False
        

        # if hands detected, multi_handedness aligns with multi_hand_landmarks order -> zip them
        if res and res.multi_hand_landmarks and res.multi_handedness:
            for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                # orientation using your function (exact)
                orientation = get_orientation(hand_landmarks, label)
                front_flag = 1 if orientation == "Front" else 0

                # compute finger states
                states = compute_finger_states(hand_landmarks, w, h)

                # assign to left/right arrays using label
                if label == "Left":
                    left_is = True
                    finger_state_left = states
                    left_hand_front = front_flag
                elif label == "Right":
                    right_is = True
                    finger_state_right = states
                    right_hand_front = front_flag
                else:
                    # fallback: choose by mean x position
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    cx = int(np.mean(xs) * w)
                    if cx < w//2:
                        finger_state_left = states
                        left_hand_front = front_flag
                    else:
                        finger_state_right = states
                        right_hand_front = front_flag

                # optional: draw landmarks & label (comment out if you prefer no overlay)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) if DEBUG else None
                # wrist text
                wr = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wx, wy = int(wr.x * w), int(wr.y * h)
                draw_text(frame, f"{label} {orientation}", wx - 60, wy - 20) if DEBUG else None

        # example if/else using produced variables
        left_open_count = sum(finger_state_left)
        right_open_count = sum(finger_state_right)

        # simple example actions (replace with your own logic)
        if not end:
            if finger_state_left[2:] == [0,0,0] and left_is and not right_is and left_hand_front==1:
                action = "Flame On"
                frame = flame.process_frame(frame,hand_landmarks=  hand_landmarks, hand_handedness= "left" )
            
            elif right_is and left_is and finger_state_right[2:] == [0,0,0] and finger_state_left[2:] == [0,0,0]:
                action = "TORNADOO!"
                frame = tornado.process_frame(frame_bgr=frame, hand_landmarks_list=res.multi_hand_landmarks)
            elif right_is and left_is:
                action = "Thunder Down"
                frame = thunder.process_frame(frame=frame, hand_landmarks_list=res.multi_hand_landmarks)
            
            elif right_is:
                action = "Water Up"
                frame = water.process_frame(frame= frame, hand_landmarks=hand_landmarks)
            elif left_is and left_hand_front == 0:
                action = "BRING ME THANOS!"
                frame = gauntlet.process_frame(frame=frame, hand_landmarks=hand_landmarks)
                end = gauntlet.ending
                
            
            else:
                action = "No Action Yet"
        else:
            action = "No Turning Back"
            frame = gauntlet.process_frame(frame=frame, hand_landmarks=hand_landmarks)

        # draw results
        if DEBUG: 
            draw_text(frame, f"Left: {finger_state_left} front:{left_hand_front}", 8, 22)
            draw_text(frame, f"Right:{finger_state_right} front:{right_hand_front}", 8, 50)
            draw_text(frame, f"Action: {action}", 8, 80)

        cv2.imshow("Finger states (ESC to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("v"):
            DEBUG = not DEBUG

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
