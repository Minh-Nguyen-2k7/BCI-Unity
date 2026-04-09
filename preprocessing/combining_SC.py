import numpy as np
import glob
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

print("Running code...")

# 1. Modified loader to track subject IDs
def get_combined_data_with_groups(path):
    x_files = sorted(glob.glob(f'{path}/X_sub_*.npy'))
    y_files = sorted(glob.glob(f'{path}/y_sub_*.npy'))

    x_list, y_list, group_list = [], [], []

    for i, (xf, yf) in enumerate(zip(x_files, y_files)):
        x_data = np.load(xf)
        y_data = np.load(yf)

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

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 64)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 64)).reshape(X_test.shape)

# Save the files
np.save(f'{save_path}/X_train_final_SC.npy', X_train_scaled)
np.save(f'{save_path}/y_train_final_SC.npy', y_train)
np.save(f'{save_path}/X_test_final_SC.npy', X_test_scaled)
np.save(f'{save_path}/y_test_final_SC.npy', y_test)

print('✅ Data saved with GroupShuffleSplit!')