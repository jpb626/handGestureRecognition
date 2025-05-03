#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from sklearn.base import TransformerMixin, BaseEstimator

# copy over the scaling class so we can load the pipeline right
class HuScaler(BaseEstimator, TransformerMixin):
    """ Must match exactly the definition used when training. """
    def __init__(self, w_hu=1.0):
        self.w_hu = w_hu
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_hog = X[:, :-7]
        X_hu  = X[:,  -7:] * self.w_hu
        return np.hstack([X_hog, X_hu])


# Load the saved pipeline
print("Loading pipeline…")
pipeline = load('final_model.joblib')
print("Pipeline loaded.")

# HOG setup (match training setup)
win_size     = (640, 480)
cell_size    = (80, 80)
block_size   = (cell_size[0]*2, cell_size[1]*2)
block_stride = cell_size
nbins        = 6

hog = cv2.HOGDescriptor(
    _winSize     = win_size,
    _blockSize   = block_size,
    _blockStride = block_stride,
    _cellSize    = cell_size,
    _nbins       = nbins
)

# mediapipe setup
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
            return mask

        # 1) draw skeleton lines
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

# Gesture label map (must match the training label setup)
gesture_names = {
    0: 'palm',
    1: 'thumb',
    2: 'index',
    3: 'ok',
    4: 'c'
}

def main():
    print("Opening webcam…")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return


    with mp_hands.Hands(
         static_image_mode=False,
         max_num_hands=1,
         min_detection_confidence=0.6,
         min_tracking_confidence=0.6) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            frame = cv2.resize(frame, win_size)
                

            # a) Segment
            mask = extract_hand_skeleton_mask(frame, hands)

            # b) Resize & extract features
            mask_resized = cv2.resize(mask, win_size)
            hog_vec      = hog.compute(mask_resized).ravel()
            m            = cv2.moments(mask_resized)
            hu           = cv2.HuMoments(m).ravel()
            feat_raw     = np.hstack([hog_vec, hu]).reshape(1, -1)

            # c) Predict with your pipeline
            pred = pipeline.predict(feat_raw)[0]
            label = gesture_names.get(pred, str(pred))

            # d) Overlay
            cv2.putText(frame,
                        f"Gesture: {label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 0), 
                        2)

            # e) Display
            cv2.imshow("Live", frame)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
