import numpy as np
import glob
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import GroupShuffleSplit


print("Running code...")

def euclidean_alignment(X):
    """
    Apply Euclidean Alignment to EEG trials for a single subject.

    For each trial, computes the covariance matrix, averages them across
    all trials, then whitens every trial with the inverse square root of
    that mean covariance matrix (He & Wu, 2019).

    Args:
        X: ndarray of shape (n_epochs, n_channels, n_timepoints)

    Returns:
        X_aligned: ndarray of the same shape, EA-transformed
    """
    n_epochs, n_channels, n_times = X.shape

    # Compute per-trial covariance matrices: (n_epochs, n_channels, n_channels)
    covs = np.stack([
        (X[i] @ X[i].T) / n_times for i in range(n_epochs)
    ])

    # Mean covariance across all trials for this subject
    R_mean = covs.mean(axis=0)

    # Inverse square root of the mean covariance (.real strips numerical noise from eigendecomposition)
    R_inv_sqrt = fractional_matrix_power(R_mean, -0.5).real

    # Whiten each trial: X_aligned[i] = R^{-1/2} @ X[i]
    X_aligned = np.stack([R_inv_sqrt @ X[i] for i in range(n_epochs)])

    return X_aligned

# 1. Modified loader to track subject IDs
def get_combined_data_with_groups(path):
    x_files = sorted(glob.glob(f'{path}/X_sub_*.npy'))
    y_files = sorted(glob.glob(f'{path}/y_sub_*.npy'))

    x_list, y_list, group_list = [], [], []

    for i, (xf, yf) in enumerate(zip(x_files, y_files)):
        x_data = np.load(xf)
        y_data = np.load(yf)

        # Apply Euclidean Alignment per subject before pooling
        x_data = euclidean_alignment(x_data)

        # Create a group ID for this subject (e.g., all epochs for Sub_1 = 0, Sub_2 = 1)
        groups = np.full(len(y_data), i)

        x_list.append(x_data)
        y_list.append(y_data)
        group_list.append(groups)

    return np.concatenate(x_list), np.concatenate(y_list), np.concatenate(group_list)

# Load data and the new group labels
save_path = r'C:\Users\vient\OneDrive\Documents\BCI-UNITY\EEGDATA\processed_npy'
X, y, groups = get_combined_data_with_groups(save_path)

# 2. Use GroupShuffleSplit instead of train_test_split
# This ensures that 16 subjects go to train and 4 subjects go to test (80/20 split)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"✅ Split Complete! Train Subjects: {len(np.unique(groups[train_idx]))}, Test Subjects: {len(np.unique(groups[test_idx]))}")

# After your split, add these two lines
subject_ids_train = groups[train_idx]
subject_ids_test = groups[test_idx]
    
# Save the files
np.save(f'{save_path}/X_train_final_EA.npy', X_train)
np.save(f'{save_path}/y_train_final_EA.npy', y_train)
np.save(f'{save_path}/X_test_final_EA.npy', X_test)
np.save(f'{save_path}/y_test_final_EA.npy', y_test)
np.save(f'{save_path}/subject_ids_train_final_EA.npy', subject_ids_train)
np.save(f'{save_path}/subject_ids_test_final_EA.npy', subject_ids_test)

print('✅ Data saved with GroupShuffleSplit!')
# Verify
print(f"Subject IDs in train: {np.unique(subject_ids_train)}")
print(f"Subject IDs in test:  {np.unique(subject_ids_test)}")