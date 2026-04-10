# BCI Motor Imagery — Real-Time Unity Control

> Translating EEG brain signals into real-time object movement in Unity Engine using a subject-independent deep learning model.

## Overview

This project implements a Brain-Computer Interface (BCI) pipeline that classifies EEG motor imagery signals into three states — **Rest**, **Left Hand**, and **Right Hand** — and uses those predictions to control a rotating object in Unity in real time.

The pipeline covers everything from raw EEG data processing, model training, ONNX export, to Unity inference using the Inference Engine package. The current implementation simulates a live headset by streaming pre-recorded EEG test data from a CSV file.

**Current Status:** Working end-to-end simulation. Real headset integration is a planned future step.

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | PhysioNet EEG Motor Imagery |
| Subjects | 20 (16 train, 4 test) |
| Classes | Rest, Left Hand, Right Hand |
| Model | EEGNet |
| Test Accuracy | 54.21% (on unseen subjects) |

## Pipeline Overview

```
Raw EEG Data (PhysioNet .edf files)
        ↓
Preprocessing (MNE, Bandpass Filter, StandardScaler)
        ↓
Model Training (EEGNet, PyTorch, Kaggle T4 GPU)
        ↓
ONNX Export → Embedded → Serialized to .sentis
        ↓
Unity Inference Engine (Inference Engine 2.2.2)
        ↓
Real-time Object Control (Rotate Left / Right / Rest)
```

## Repository Structure

```
BCI-Motor-Imagery-Real-time-Unity-Control/
    notebooks/
        EEG.ipynb              # Attempt 1 — Data checkpoint
        EEG_ver0.ipynb         # Attempt 2 — Logistic Regression, Random Forest, SVM
        EEG_ver1.ipynb         # Attempt 3 — Adjusted data conversion
        eegonkaggle.ipynb      # Attempt 4 — EEGNet introduction
        final-model.ipynb      # Attempt 5 — Final EEGNet configuration + ONNX export
    preprocessing/
        epoching.py            # Downloads PhysioNet data via MNE, creates per-subject epochs
        combining_SC.py        # Combines subjects, applies StandardScaler, splits train/test
        combining_EA.py        # Experimental — Euclidean Alignment approach (abandoned)
        generate_csv.py        # Converts test .npy files into Unity-readable CSV format
    unity/
        EEGMODEL.cs            # Model inference script
        EEGStreamer.cs         # CSV streaming script
        RotateObject.cs        # Object movement script
    models/
        eegnet_embedded.onnx   # Exported model with embedded weights
    README.md
    JOURNEY.md
```

## Requirements

### Python / Kaggle
- Python 3.12
- PyTorch
- MNE
- NumPy
- scikit-learn
- ONNX
- ONNX Runtime

### Unity
- Unity 6.3 LTS (6000.3.1f1)
- Unity Inference Engine 2.2.2

Install Inference Engine by adding this to `Packages/manifest.json`:
```json
"com.unity.sentis": "2.1.1"
```
This will install as **Inference Engine 2.2.2** — the invalid signature warning that appears is safe to ignore.

## How to Replicate

### 1. Data & Preprocessing
- Run `preprocessing/epoching.py` to download PhysioNet data via MNE and create per-subject epoch files
- Run `preprocessing/combining_SC.py` to combine all subjects, apply StandardScaler normalization, and generate final train/test `.npy` files
- Run `preprocessing/generate_csv.py` to convert the test `.npy` files into `eeg_test_stream.csv` for Unity streaming
- Note: `preprocessing/combining_EA.py` is an experimental Euclidean Alignment approach that was tested but abandoned due to accuracy drops

### 2. Model Training & ONNX Export
- Upload the generated `.npy` files to Kaggle as a dataset
- Run `notebooks/final-model.ipynb` on Kaggle using a T4 GPU
- The notebook will automatically export the model to ONNX and embed the weights
- Download `eegnet_embedded.onnx` from the Kaggle output

### 3. Preparing the Model for Unity
- Place `eegnet_embedded.onnx` in your Unity project's `Assets/Models/` folder
- In Unity's Inspector, click **"Serialize To StreamingAssets"**
- This generates `eegnet_embedded.sentis` in the `StreamingAssets/` folder
- Note: Unity Inference Engine cannot load `.onnx` directly — the `.sentis` serialization step is required

### 4. Unity Setup
- Open a new Unity 6.3 LTS project
- Install Inference Engine via `manifest.json`
- Add the three C# scripts from `unity/` to your scene
- Place `eeg_test_stream.csv` in `StreamingAssets/`
- Assign script references in the Inspector:
  - `EEGStreamer` → assign the GameObject with `EEGMODEL` as Classifier
  - `RotateObject` → assign the GameObject with `EEGStreamer` as Streamer

### 5. Running
- Press Play in Unity
- The cube will rotate based on EEG predictions from the CSV stream
- Console will show epoch predictions and running accuracy

## Known Issues & Notes
- Unity Inference Engine **must** load the `.sentis` file, not the `.onnx` file directly — loading `.onnx` results in 0MB weight size and incorrect predictions
- Use `BackendType.CPU` — the `GPUCompute` backend produces incorrect results with depthwise convolutions
- The `com.unity.sentis` package installs as **Inference Engine 2.2.2** in Unity 6 — this is expected
