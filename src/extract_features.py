import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_feature_matrix(data_tuples, return_raw_and_scaler=False, verbose=True):
    """
    Args:
      data_tuples: list of (img_path, label, subject_id)
      verbose    : if True, prints progress
    Returns:
      X      : ndarray, shape (N, D) feature matrix
      y      : ndarray, shape (N,) integer labels
      groups : ndarray, shape (N,) subject IDs for LOSO grouping
    """
    # HOG parameters - choice of 80 cell size because it balances # of dims with detail while 
    # remaining a multiple of both 640 and 480 (the image size).
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

    feats, labels, groups = [], [], []
    total = len(data_tuples)
    current = 0
    last_subject = None
    subject_counts = {}

    for img_path, label, subject in data_tuples:
        current += 1
        # Print per-image progress
        if verbose:
            print(f"[{current:4d}/{total}] Subject {subject:02d}  Label {label}  → {img_path}")

        # Track counts per subject
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

        # Load & ensure correct size / gray
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Couldn’t read {img_path}")
        if img.shape[::-1] != win_size:
            img = cv2.resize(img, win_size)

        # HOG Descriptors
        hog_vec = hog.compute(img).ravel()
        
        # Hu moments
        m  = cv2.moments(img)
        hu = cv2.HuMoments(m).ravel()

        # Concatenate features
        feat = np.hstack([hog_vec, hu])
        feats.append(feat)
        # feats.append(hu)
        labels.append(label)
        groups.append(subject)

    # Print per-subject summary
    if verbose:
        print("\nSamples per subject:")
        for subj, cnt in sorted(subject_counts.items()):
            print(f"  Subject {subj:02d}: {cnt} images")

    #  Stack data into arrays
    X_raw  = np.vstack(feats)
    y      = np.array(labels, dtype=int)
    groups = np.array(groups, dtype=int)


    scaler = StandardScaler().fit(X_raw)
    if return_raw_and_scaler:
        return X_raw, y, groups, scaler
    X = scaler.transform(X_raw)
    return X, y, groups


if __name__ == '__main__':
    from loader import make_data_tuples

    # Compute data_tuples,  can adjust path as needed
    data_tuples = make_data_tuples('../segmented-data/leapGestRecog')

    X_raw, y, groups, scaler = build_feature_matrix(data_tuples, return_raw_and_scaler=True)
    X_scaled = scaler.transform(X_raw)

    np.savez_compressed(
        'final_features_80x80_raw_and_scaled.npz',
        X_raw=X_raw,
        X=X_scaled,
        y=y,
        groups=groups
    )
    print("Saved final_features_80x80_raw_and_scaled.npz")

    print("Feature matrix:", X_raw.shape)
    print("Labels:", np.unique(y), "Subjects:", np.unique(groups))



    # to load these features latyer:
    # data = np.load('features.npz')
    # X = data['X']
    # y = data['y']
    # groups = data['groups']
