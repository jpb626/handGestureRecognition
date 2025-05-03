#!/usr/bin/env python3
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands solution.
mp_hands = mp.solutions.hands
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
  

def process_image_file(image_path, out_path, hands):
    """
    Reads an image, resizes it to 640x480, runs the segmentation algorithm,
    and writes the resulting binary mask back to the same file.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    # Resize the image to 640x480.
    resized_image = cv2.resize(image, (640, 480))
    
    # Get the binary hand skeleton mask.
    mask = extract_hand_skeleton_mask(resized_image, hands)
    if mask is None:
        if os.path.exists(out_path):
            os.remove(out_path)
        return False

    # Ensure save directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Overwrite the original image with the binary mask.
    success = cv2.imwrite(out_path, mask)

    return success


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    # Base directory for the dataset.
    base_dir = os.path.join(HERE, "..", "..", "data", "leapGestRecog")
    save_dir = os.path.join(HERE, "..", "..", "segmented-data", "leapGestRecog")
    # Subject folders are "00" to "09".
    subjects = [f"{i:02d}" for i in range(10)]
    # Image folders to process.
    image_folders = ["01_palm", "03_fist", "05_thumb", "06_index", "07_ok", "09_c"]
    
    # Initialize Mediapipe Hands in static image mode.
    with mp_hands.Hands(
         min_detection_confidence=0.2,
         min_tracking_confidence=0.2,
         max_num_hands=1,
         static_image_mode=True) as hands:
        
        # Loop through each subject folder.
        for subject in subjects:
            for folder in image_folders:
                in_folder  = os.path.join(base_dir, subject, folder)
                out_folder = os.path.join(save_dir, subject, folder)

                if not os.path.isdir(in_folder):
                    print(f"Folder not found: {in_folder}")
                    continue

                # Process each PNG file in the folder.
                for filename in os.listdir(in_folder):
                    if not filename.lower().endswith(".png"):
                        continue
                    in_path = os.path.join(in_folder, filename)
                    out_path = os.path.join(out_folder, filename)

                    success = process_image_file(in_path, out_path, hands)
                    if success:
                        print(f"Processed: {in_path}")
                    else:
                        print(f"Failed: {in_path}")

if __name__ == '__main__':
    main()
