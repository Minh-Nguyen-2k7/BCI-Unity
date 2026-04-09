import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import os

# Your local directory from the screenshot
base_path = r'C:\Users\vient\OneDrive\Documents\BCI-UNITY\EEGDATA'
save_path = os.path.join(base_path, 'processed_npy')
os.makedirs(save_path, exist_ok=True)

# Choose your subjects (S001 to S010)
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Runs 3, 7, 11 = Task 1 (Opening/closing left or right fist)
runs = [3, 7, 11]

all_epochs = []
real_events = []
all_X = []
all_y = []

# Run for each subject
for sub in subjects:
    # 1. Fetch and load 3 EEG files (runs) into raw
    files = eegbci.load_data(sub, runs)
    raws = [read_raw_edf(f, preload=True, stim_channel='auto') for f in files]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    # Keep ONLY the 64 EEG channels and remove the STIM channel
    raw.pick_types(eeg=True, stim=False, exclude='bads')

    # Notch + Bandpass + CAR
    raw.notch_filter(60)
    raw.filter(l_freq=8, h_freq=30)
    raw.set_eeg_reference(ref_channels='average', projection=False)

    # 2. Extract the actual event markers in raw
    # T0=0 (Rest), T1=1 (Left), T2=2 (Right)
    real_events, _ = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
    print(f"Real Events Shape: {real_events.shape}") # (90, 3) because 30 trials each file, with 3 unique event labels

    # 3. Create empty markers throughout raw
    # This places a marker every 0.1s (6.25 samples) for the duration of the raw data
    stamps = mne.make_fixed_length_events(raw, id=999, duration=0.1, first_samp=True)
    print(f"Stamps Shape: {stamps.shape}")

    # 4. Assign empty markers based on real_events marker
    all_labeled_stamps = []
    sfreq = raw.info['sfreq']
    trial_duration_samples = int(4.1 * sfreq)

    for start_samp, _, label in real_events:
      # Assign stamps based on window size (start to start + 4s)
      end_samp = start_samp + trial_duration_samples

      # Select stamps inside this specific 4s window
      mask = (stamps[:, 0] >= start_samp) & (stamps[:, 0] < end_samp)
      matched_stamps = stamps[mask].copy()

      # Change ID from 999 to the real_event label
      matched_stamps[:, 2] = label

      all_labeled_stamps.append(matched_stamps)

    # Combine all of the stamps into a usual y_train/test labels
    final_events = np.vstack(all_labeled_stamps)
    # Create the 0.6s windows. Each epoch has size of 97 samples.
    # For each trial we take 4.1s, so we have 41 stamps. Ideally we would have a total of 3690 epochs (stamps x trials), but some epochs probably does not have the size for an epoch.
    epochs = mne.Epochs(raw, events=final_events,
                    event_id={'rest': 0, 'left': 1, 'right': 2},
                    tmin=0, tmax=0.8, baseline=None, preload=True)

    # After creating epochs for THIS subject, we get the X and y, and append.
    X_sub = epochs.get_data()
    y_sub = epochs.events[:, -1]
    subject_ids_sub = np.full(len(y_sub), sub - 1) # Creating subject IDs (0-indexed)

    # Save to your hard drive
    np.save(f'{save_path}/X_sub_{sub}.npy', X_sub)
    np.save(f'{save_path}/y_sub_{sub}.npy', y_sub)
    np.save(f'{save_path}/subject_ids_sub_{sub}.npy', subject_ids_sub)
  
print("✅ All subjects saved as .npy files!")
