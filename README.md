
# Hand Gesture Recognition Project

This repository implements a complete pipeline for real‑time hand gesture recognition, from raw video frames through segmentation, feature extraction, classifier training, and live inference. It uses Google’s MediaPipe Hands for landmark detection as a base for the hand segmentation algorithm, classical HOG + Hu moment features, and a linear SVM trained with leave‑one‑subject‑out cross‑validation. The system achieves ≥15 FPS on standard hardware.

---

## Table of Contents

- [Features](#features)  
- [Requirements & Installation](#requirements--installation)  
- [1. Features](#1-features)  
- [2. Requirements & Installation](#2-requirements--installation)  
- [3. Dataset Prep](#3-dataset-prep)  
- [4. Extracting features](#4-extracting-features)  
- [5. Training the model](#5-training-the-model)  
- [Output Artifacts](#output-artifacts)  
- [6. Using the model](#6-using-the-model)

---

## Features

- Real‑time segmentation with MediaPipe Hands (21 landmarks)  
- Binary mask generation (skeleton drawing + palm filling)  
- HOG descriptors + Hu invariant moments feature extraction  
- Linear SVM classifier (one‑versus‑one)  
- Leave‑one‑subject‑out cross‑validation  
- Live inference on webcam or video files at ≥15 FPS  
- Standalone scripts for every stage  

---

## Requirements & Installation

```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

# Create environment
conda env create -f environment.yml
conda activate project_env

---

## 1. Features

- **Real‑time segmentation** with MediaPipe Hands (21‑landmark detector)  
- **Binary mask generation** via skeleton drawing + palm filling  
- **Robust feature extraction** using HOG descriptors and Hu invariant moments  
- **Linear SVM classifier** in a one‑versus‑one configuration  
- **Leave‑one‑subject‑out cross‑validation** for strong generalization  
- **Live inference** on webcam or video files at ≥15 FPS  
- **Easy-to-use scripts** for data prep, training, and inference  

---

## 2. Requirements & Installation

1. **Clone this repository**  

   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Create / Activate Environment**  

    ```bash
    conda env create -f environment.yml
    conda activate project_env
    ```

---

## 3. Dataset Prep

The following files are all in src/segmentation/ folder. The Leap Gesture Recognition dataset (Mantecon2016) is already downloaded inside the zip or github repository. The segmented version of the dataset is also ready, but can be run using the following commands from the root directory:

 ```bash
 cd src/segmentation/
 python segment-data.py
 ```

segment-data.py will go through each subject in the data/leapGestRecog/ unprocessed dataset, apply the deep learned mediapipe hand landmark algorithm to get the hand mask of each image, filling in the palm to add shaping to the hand.
It will then save the resulting segmented image into an equivalently named place inside segmented-data/leapGestRecog/
The main function in segment-data.py specifies important parameters that can be played with including the input and output folders, the mediapipe hand landmark confidence (the larger the less data will make it through the segmentation process but what does get through will likely be of higher quality), and what image folders you want to be segmented from data/leapGestRecog/ . Our final model uses "01_palm", "05_thumb", "06_index", "07_ok", "09_c".

 python segment-video.py

segment-video.py can also be run if you'd like to see the live segmentation on your webcam.

 python get_more_data.py 01_palm

get_more_data.py can also be run if you'd like to add more subjects to the dataset. It takes video feed from the webcam, segments it according to the algorithm previously described (most verbosely in Latex/Report.pdf) and takes 10 frames per second from your video feed and sorts your hand gesture into whichever folder you enter with the command. The example command sets this folder to 01_palm, so by default the resulting frames will be stored in segmented-data/leapGestRecog/10/01_palm. and each frame will be stored as frame_10_01_palm_0001.png where the last value "0001" is the frame number of the image and "10" is the specified subject number.

---

## 4. Extracting features

Make sure your in the src/ directory for the feature commands

 python extract_features.py

extract_features.py reads the list of data tuples from loader.py, loads each grayscale mask, resizes it to 640x480, computes the HOG descriptor and Hu moments, concatenates them into a feature vector, and stacks all vectors into X. It also fits a StandardScaler on the raw features and saves both X and the scaler (plus labels and subject IDs) into a compressed features.npz for fast downstream use. It's recommended changing the name of the outputted features.npz file according to the changes to the descriptor specifications you may change inside extract_features.py so that you do not override old compressed feature files.
Our final model's used features are saved as features_80x80_raw_and_scaled.npz.

---

## 5. Training the model

Make sure your in the src/ directory

 python train_svm.py

Loads the precomputed features.npz, splits off one subject for final testing, and uses leave-one-subject-out cross-validation on the remaining subjects to select the SVM's regularization parameter C via GridSearchCV. It then retrains the best estimator on all training subjects, evaluates on the held-out test subject, saves the confusion-matrix plots and the final .joblib model, and logs each run's parameters and accuracies to results.csv. The csv file can be easily initialized by running:
!WARNING! YOU MUST RUN THE BELOW BEFORE TRAINING IF YOU WANT TO EASILY STORE RESULTS:

 python create_results_csv.py

Our final model is saved as final_model.joblib, to get an equivalent model using this program, make sure to load features_80x80_raw_and_scaled.npz at the beginning of the file.
Our AB testing experimental models were ran using this file and saved inside results_table.csv

Note: train_svm.py does have some experimental code to scale the Hu moments up to match the size of the HOG descriptors but this scaling is not actually done in this file as I chose not to implement this scaling due to experimental failures.

### Output Artifacts

After running the training script, you’ll find everything in `output/`:

- **Cross‑Validation Metrics**  
  Prints CV accuracy and per‑class precision/recall/F1 (via `classification_report`), then saves `CM-CV.png`—a confusion matrix heatmap showing where gestures get confused during leave‑one‑subject‑out CV.

- **Test‑Set Metrics**  
  Prints test accuracy on the held‑out subject plus its classification report, and saves `CM-Test.png` for the test confusion matrix.

- **ROC & PR Curves**  
  Generates `ROC_Test.png` (ROC curves with AUC per class) and `PR_Test.png` (precision‑recall curves with Average Precision), giving you a threshold‑independent view of each gesture’s separability.

- **Serialized Model**  
  Saves the full pipeline (HOG+Hu+SVM) to `svm_model_C{best_C}.joblib` so you can reload it for inference without retraining.

- **Results Log**  
  Appends a line to `results.csv` with your key parameters (HOG cell size, Hu scaling, SVM C, etc.) and the CV/test accuracies—ideal for tracking and comparing multiple runs.

## 6. Using the model

Make sure your in the src/ directory

 python get_gesture_video.py

get_gesture_video.py captures live video from your webcam, applies the segmentation algorithm masking your hand, displaying the resulting binary mask. It then loads the trained SVM model and uses it to classify the gesture in real time. The script also displays the predicted gesture label on the video feed.
The final model is loaded as final_model.joblib
