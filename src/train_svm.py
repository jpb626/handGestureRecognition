#!/usr/bin/env python
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)

# --- 1) HuScaler transformer ---
class HuScaler(BaseEstimator, TransformerMixin):
    """ Multiply the last 7 Hu‐moment columns by w_hu. """
    def __init__(self, w_hu=1.0):
        self.w_hu = w_hu
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_hog = X[:, :-7]
        X_hu  = X[:,  -7:] * self.w_hu
        return np.hstack([X_hog, X_hu])

# --- 2) Hyperparameter grid ---
param_grid = {'svc__C': [1e-3,1e-2,1e-1,1,10,100,1000]}

# --- 3) CSV metadata ---
csv_parameters = {
    "kernel_type":             "linear",
    "decision_function_shape": "ovo",
    "HOG Cell Size":           "80x80",
    "Hu Moments Scaled Up":    True,
    "HOG Disabled":            False
}

# --- 4) Configuration ---
KERNEL      = 'linear'    # or 'rbf'
PROBABILITY = True        # enable probability for ROC/PR
SHAPE       = 'ovo'       # 'ovo' or 'ovr'

# --- 5) Load raw+scaled features ---
data = np.load('features_80x80_raw_and_scaled.npz')
X_raw     = data['X_raw']      # unscaled HOG+Hu
y_all     = data['y']
groups_all= data['groups']

# carve out subject 9 for final test
test_subject = 9
mask_test    = (groups_all == test_subject)
mask_train   = ~mask_test

Xr_train, y_train, g_train = X_raw[mask_train], y_all[mask_train], groups_all[mask_train]
Xr_test,  y_test,  g_test  = X_raw[mask_test],  y_all[mask_test],  groups_all[mask_test]

print(f"Train on subjects: {sorted(set(g_train))}")
print(f"Hold-out test subject: {test_subject} ({len(y_test)} samples)")

# --- 6) Compute Hu‐weight ---
D       = X_raw.shape[1]
var_hog = X_raw[:, :D-7].var(axis=0).sum()
var_hu  = X_raw[:,  D-7:].var(axis=0).sum()
w_hu    = np.sqrt(var_hog / var_hu)
print(f"Scaling Hu moments by w_hu = {w_hu:.2f}")

# --- 7) Build pipeline ---
pipe = Pipeline([
    ('hu_scale', HuScaler(w_hu=w_hu)),
    ('scale',     StandardScaler()),
    ('svc',       SVC(kernel=KERNEL,
                      decision_function_shape=SHAPE,
                      probability=PROBABILITY))
])

logo = LeaveOneGroupOut()
grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=logo.split(Xr_train, y_train, g_train),
    scoring='accuracy',
    n_jobs=2,
    refit=True
)
print(f"\nRunning GridSearchCV (kernel={KERNEL}, shape={SHAPE}) …")
grid.fit(Xr_train, y_train)

best_C    = grid.best_params_['svc__C']
best_pipe = grid.best_estimator_

print(f"Best C: {best_C}")
print(f"Best LOSO-CV score: {grid.best_score_*100:.2f}%")

# --- 8) CV predictions & metrics ---
y_pred_cv = cross_val_predict(
    best_pipe, Xr_train, y_train,
    groups=g_train,
    cv=logo,
    method='predict'
)
acc_cv = accuracy_score(y_train, y_pred_cv)
print(f"\nRecomputed CV Accuracy: {acc_cv*100:.2f}%")
print("\nClassification Report (CV):")
print(classification_report(y_train, y_pred_cv))

cm_cv  = confusion_matrix(y_train, y_pred_cv, labels=np.unique(y_train))

# --- 9) Test metrics ---
y_pred_test = best_pipe.predict(Xr_test)
acc_test    = accuracy_score(y_test, y_pred_test)
print(f"\nTest Accuracy on subject {test_subject}: {acc_test*100:.2f}%")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

cm_test     = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_test))

# prepare output folder
output_dir = os.path.join('..', 'output')
os.makedirs(output_dir, exist_ok=True)

# Plot and save confusion matrices
for cm, split in [(cm_cv, 'CV'), (cm_test, 'Test')]:
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_train if split=='CV' else y_test))
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'{split} Confusion Matrix (C={best_C})')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'CM-{split}.png'))
    plt.close(fig)

# --- 10) ROC and PR curves (Test set) ---
classes = np.unique(y_test)
y_test_b = label_binarize(y_test, classes=classes)
y_score  = best_pipe.decision_function(Xr_test)

# ROC curves
plt.figure()
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Test)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'ROC_Test.png'))
plt.close()

# Precision-Recall curves
plt.figure()
for i, cls in enumerate(classes):
    precision, recall, _ = precision_recall_curve(y_test_b[:, i], y_score[:, i])
    ap = average_precision_score(y_test_b[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {cls} (AP = {ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (Test)')
plt.legend(loc='lower left')
plt.savefig(os.path.join(output_dir, 'PR_Test.png'))
plt.close()

# --- 11) Save model & log CSV ---
model_filename = os.path.join(output_dir, f'svm_model_C{best_C}.joblib')
dump(best_pipe, model_filename)

out_csv = os.path.join(output_dir, 'results.csv')
row = [
    csv_parameters['kernel_type'],
    csv_parameters['decision_function_shape'],
    csv_parameters['HOG Cell Size'],
    csv_parameters['Hu Moments Scaled Up'],
    csv_parameters['HOG Disabled'],
    best_C,
    f"{acc_cv*100:.2f}%",
    f"{acc_test*100:.2f}%"
]
with open(out_csv, 'a', newline='') as f:
    csv.writer(f).writerow(row)

print(f"\nSaved model to {model_filename}")
print(f"Results logged to {out_csv}")

