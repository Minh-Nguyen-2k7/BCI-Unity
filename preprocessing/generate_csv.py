import numpy as np
import pandas as pd

# Load test data
X_test = np.load('X_test_final_SC.npy')
y_test = np.load('y_test_final_SC.npy')

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Flatten each epoch from (64, 129) to (8256,) per row
n_samples = X_test.shape[0]
X_flat = X_test.reshape(n_samples, -1)

# Add label column at the end
labels = y_test.reshape(-1, 1)
data = np.hstack([X_flat, labels])

# Create column names
channel_names = [f'ch{c}_t{t}' for c in range(64) for t in range(129)]
columns = channel_names + ['label']

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('eeg_test_stream.csv', index=False)

print(f"✅ Saved! Shape: {df.shape}")
print(f"Columns: {len(columns)} total")