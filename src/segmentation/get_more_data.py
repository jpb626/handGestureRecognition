#!/usr/bin/env python3
import os
import cv2
import mediapipe as mp
import numpy as np
import time


# Change this to the subject's ID (here “10”) that we want to record data for
SUBJECT_ID = "10"

# Pass your desired gesture folder (e.g. "01_palm") as a command‑line arg
import sys
if len(sys.argv) < 2:
    print("Usage: python record_gesture.py <folder_name>")
    sys.exit(1)
GESTURE_FOLDER = sys.argv[1]

# Build the output path: ../../segmented-data/leapGestRecog/10/01_palm
OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "segmented-data", "leapGestRecog",
    SUBJECT_ID, GESTURE_FOLDER
)
os.makedirs(OUT_DIR, exist_ok=True)

# Mediapipe setup
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_skeleton_mask(image, hands):
    """
    Returns a binary mask: fingers as skeleton lines + filled palm.
    """
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if not results.multi_hand_landmarks:
            return None

        # draw skeleton lines
        lm = results.multi_hand_landmarks[0]
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        mp_drawing.draw_landmarks(
            tmp, lm, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
        )
        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        _, skel = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, skel)

        # build palm polygon
        PALM_IDX = [0, 5, 9, 13, 17]
        pts = []
        for idx in PALM_IDX:
            lm_pt = lm.landmark[idx]
            x, y  = int(lm_pt.x * w), int(lm_pt.y * h)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32).reshape(-1,1,2)

        # fill & smooth palm
        palm_mask = np.zeros_like(mask)
        cv2.fillPoly(palm_mask, [pts], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        palm_mask = cv2.morphologyEx(palm_mask, cv2.MORPH_CLOSE, kernel)

        # merge
        mask = cv2.bitwise_or(mask, palm_mask)

        # final binarize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # final dilation
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    except Exception as e:
        print("ERROR in extract_hand_skeleton_mask:", e)
        # return something safe so the loop can continue
        return np.zeros(image.shape[:2], dtype=np.uint8)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    with mp_hands.Hands(
         static_image_mode=False,
         max_num_hands=1,
         min_detection_confidence=0.6,
         min_tracking_confidence=0.6) as hands:

        frame_idx = 1
        last_save = 0.0
        print(f"Recording to {OUT_DIR} … press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame from webcam")
                break

            frame = cv2.resize(frame, (640, 480))
            mask = extract_hand_skeleton_mask(frame, hands)

            # save at ~10 fps
            now = time.time()
            if now - last_save >= 0.1:
                filename = f"frame_{SUBJECT_ID}_{GESTURE_FOLDER}_{frame_idx:04d}.png"
                outpath  = os.path.join(OUT_DIR, filename)
                cv2.imwrite(outpath, mask)
                print(f"Saved: {outpath}")
                frame_idx += 1
                last_save = now

            # show feedback
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()
