"""
REQUIREMENTS:
  pip install mediapipe opencv-python numpy requests

On first run, the hand landmark model (~8 MB) is downloaded automatically.
"""

import cv2
import numpy as np
import time
import sys
import os
import urllib.request


try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
        HandLandmarksConnections,
    )
    from mediapipe import Image as MpImage, ImageFormat
except ImportError:
    print("Missing libraries.  Run:\n  pip install mediapipe opencv-python numpy")
    sys.exit(1)


MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB) …", flush=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH,
                reporthook=lambda b, bs, t: print(
                    f"\r  {min(b*bs, t)*100//t if t>0 else 0}%", end="", flush=True))
            print("\r  Done!          ")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("Manually download from:")
            print(f"  {MODEL_URL}")
            print(f"and place as: {MODEL_PATH}")
            sys.exit(1)


CAM_W, CAM_H    = 1280, 720
BRUSH_SIZE      = 8
ERASER_RADIUS   = 55
BOARD_BG        = (30, 30, 30)

PALETTE = [
    ("Chalk",  (255, 255, 255)),
    ("Amber",  ( 50, 180, 255)),
    ("Coral",  ( 80, 100, 255)),
    ("Mint",   (130, 220, 120)),
    ("Sky",    (220, 170,  55)),
    ("Violet", (220,  90, 200)),
    ("Gold",   ( 30, 200, 215)),
]

# Landmark indices
TIP  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
PIP  = [3, 6, 10, 14, 18]   # corresponding lower joints


def fingers_up(lms):
    """lms: list of NormalizedLandmark. Returns [thumb, idx, mid, ring, pinky]."""
    up = []
    # Thumb: x-axis (works for either hand after mirror-flip)
    up.append(lms[TIP[0]].x < lms[PIP[0]].x)
    # Fingers: tip y < pip y  (higher on screen = smaller y)
    for t, p in zip(TIP[1:], PIP[1:]):
        up.append(lms[t].y < lms[p].y)
    return up

def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

 
#  DRAWING HELPERS

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

def draw_skeleton(frame, lms, w, h):
    pts = [lm_px(lm, w, h) for lm in lms]
    for s, e in CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 200, 100), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 5 if i in TIP else 3
        cv2.circle(frame, pt, r, (0, 230, 120), -1)

def draw_cursor(frame, pt, mode, color):
    if mode == "DRAW":
        cv2.circle(frame, pt, BRUSH_SIZE, color, -1)
        cv2.circle(frame, pt, BRUSH_SIZE + 2, (255, 255, 255), 1, cv2.LINE_AA)
    elif mode == "ERASE":
        cv2.circle(frame, pt, ERASER_RADIUS, (60, 80, 220), 2, cv2.LINE_AA)
        cv2.line(frame, (pt[0]-14, pt[1]), (pt[0]+14, pt[1]), (60, 80, 220), 1)
        cv2.line(frame, (pt[0], pt[1]-14), (pt[0], pt[1]+14), (60, 80, 220), 1)

def draw_toolbar(frame, color_idx, mode, fps):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 72), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, 72), (w, 72), (50, 50, 50), 1)

    # Colour swatches
    sw, pad = 44, 10
    for i, (name, col) in enumerate(PALETTE):
        x1 = pad + i * (sw + 7)
        y1, y2 = 14, 58
        if i == color_idx:
            cv2.rectangle(frame, (x1-4, y1-4), (x1+sw+4, y2+4), (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x1+sw, y2), col, -1)
        cv2.rectangle(frame, (x1, y1), (x1+sw, y2), (70, 70, 70), 1)

    # Mode pill
    mc = {"DRAW": (130,230,120), "ERASE": (60,130,230),
          "COLOR": (230,200,55), "CLEAR": (60,80,220)}.get(mode, (150,150,150))
    cv2.putText(frame, f"  {mode}", (w-200, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (w-90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    # Legend
    legend = "  ☝ Draw   ✋ Erase   ✌ Color   ✊ Clear   Q Quit"
    cv2.putText(frame, legend, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)

def draw_color_hint(frame, color_idx, alpha):
    _, col = PALETTE[color_idx]
    name   = PALETTE[color_idx][0]
    h, w   = frame.shape[:2]
    ov = frame.copy()
    cv2.putText(ov, f"Color: {name}", (w//2-100, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Drawing canvas
    canvas = np.full((fh, fw, 3), BOARD_BG, dtype=np.uint8)

    # Build HandLandmarker (new Tasks API)
    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    color_idx     = 0
    prev_pt       = None
    mode          = "IDLE"
    color_hint_ts = 0.0
    prev_ts       = time.time()
    fps           = 0.0
    color_debounce = 0.0

    print("\n✦  Virtual Blackboard — running (press Q to quit)\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_ts, 1e-6)
        prev_ts = now

        # ── MediaPipe detection (new Tasks API) ───────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, int(now * 1000))

        mode   = "IDLE"
        cur_pt = None

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]         # first hand
            up  = fingers_up(lms)
            n   = sum(up)

            idx_tip  = lm_px(lms[8],  fw, fh)
            palm_c   = lm_px(lms[0],  fw, fh)

            if n >= 4:
                # Open palm → ERASE
                mode   = "ERASE"
                cur_pt = palm_c
                prev_pt = None
                cv2.circle(canvas, palm_c, ERASER_RADIUS, BOARD_BG, -1)

            elif n == 0:
                # Fist → CLEAR
                mode    = "CLEAR"
                prev_pt = None
                canvas[:] = BOARD_BG

            elif up[1] and up[2] and not up[3] and not up[4]:
                # ✌ Two fingers → cycle color (debounced)
                if now - color_debounce > 0.6:
                    mode          = "COLOR"
                    color_idx     = (color_idx + 1) % len(PALETTE)
                    color_hint_ts = now
                    color_debounce = now
                prev_pt = None

            elif up[1] and not up[2]:
                # ☝ Index only → DRAW
                mode   = "DRAW"
                cur_pt = idx_tip
                _, col = PALETTE[color_idx]
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, cur_pt, col,
                             BRUSH_SIZE, lineType=cv2.LINE_AA)
                prev_pt = cur_pt
            else:
                prev_pt = None

            draw_skeleton(frame, lms, fw, fh)
        else:
            prev_pt = None

        # ── Compose output ────────────────────────────────────────────────────
        # Pixels that differ from board background = ink
        diff   = cv2.absdiff(canvas, np.array(BOARD_BG, dtype=np.uint8))
        mask   = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)

        output = (frame * 0.55).astype(np.uint8)       # dim camera feed
        output[mask > 0] = canvas[mask > 0]            # overlay ink on top

        # Cursor
        if cur_pt:
            _, col = PALETTE[color_idx]
            draw_cursor(output, cur_pt, mode, col)

        # Color hint fade
        if now - color_hint_ts < 1.0:
            alpha = 1.0 - (now - color_hint_ts)
            draw_color_hint(output, color_idx, alpha)

        draw_toolbar(output, color_idx, mode, fps)

        cv2.imshow("AI Virtual Blackboard", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Bye! 👋")

if __name__ == "__main__":
    main()