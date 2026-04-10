# Project Journey — BCI Motor Imagery Real-Time Unity Control

This document chronicles the full development process of this project, including all attempts, pivots, and lessons learned. It is intended to show the thinking behind each decision rather than just the final result.

---

## The Original Plan

The project was originally structured as an 8-week plan:

- **Phase 1 (Weeks 1-3):** Data cleaning, frequency filtering, and feature extraction using BrainFlow
- **Phase 2 (Weeks 4-5):** Train a Random Forest or simple Neural Network, export to ONNX
- **Phase 3 (Weeks 6-8):** Unity integration using Sentis, CSV streaming, real-time object control

In practice, the project deviated significantly from this plan. The data source changed, the model architecture changed, and the Unity package changed. What remained constant was the end goal — reading EEG signals and translating them into real-time movement in Unity.

---

## Attempt 1 — Data Checkpoint (`EEG.ipynb`)

**Goal:** Transform raw EEG files into usable training and testing data.

This was a pure data engineering checkpoint with no ML model involved. Three EEG recording files were manually downloaded and processed into a format that Python could work with. The focus was on understanding the structure of EEG data — 64 channels, time series signals, and event markers indicating Rest, Left, and Right motor imagery.

**Outcome:** Successfully created training and testing datasets. No model trained yet.

**Key lesson:** EEG data is not plug-and-play. Understanding the channel structure, sampling rate, and event markers took significant time before any ML work could begin.

---

## Attempt 2 — First ML Models (`EEG_ver0.ipynb`)

**Goal:** Train the first real classifiers on the processed EEG data.

Three models were trained using scikit-learn: Logistic Regression, Random Forest, and SVM. All three used StandardScaler normalization and were tuned using cross-validation.

**Results:**
| Model | CV Score | Test Score |
|-------|----------|------------|
| Logistic Regression | 0.597 | 0.593 |
| Random Forest | 0.566 | 0.630 |
| SVM | 0.589 | **0.667** |

**Best model: SVM at 66.7% test accuracy**

**Outcome:** Promising results but the test data came from the same subjects as training, making this an easier problem than real-world deployment.

---

## Attempt 3 — Adjusted Data Conversion (`EEG_ver1.ipynb`)

**Goal:** Improve on Attempt 2 by adjusting how the raw EEG data was converted before training.

The same three models were retrained with a modified data conversion pipeline. The changes aimed to extract more meaningful features from the EEG signals.

**Results:**
| Model | CV Score | Test Score |
|-------|----------|------------|
| Logistic Regression | 0.620 | 0.605 |
| Random Forest | 0.604 | **0.660** |
| SVM | 0.638 | 0.636 |

**Best model: Random Forest at 66.0% test accuracy**

**Outcome:** Mixed results — some models improved, others dropped slightly. The adjusted conversion helped Logistic Regression but SVM performance declined.

**Key decision:** At this point it became clear that the manual feature engineering approach had a ceiling. The models were also not being tested on truly unseen subjects, which meant the real-world accuracy would likely be much lower. This led to a fundamental pivot in approach.

---

## Attempt 4 — Switch to EEGNet (`eegonkaggle.ipynb`)

**Goal:** Abandon the manual feature engineering approach and switch to a deep learning model designed specifically for EEG data.

Two major changes were made simultaneously:

1. **New model:** EEGNet — a compact CNN architecture designed specifically for EEG classification, replacing the sklearn models entirely
2. **New data pipeline:** Instead of manually downloading files, MNE was used to programmatically download the full PhysioNet EEG Motor Imagery dataset (20 subjects, runs 3/7/11). `GroupShuffleSplit` was used to ensure the 4 test subjects were completely unseen during training — making this a true subject-independent evaluation

This was the first time the model was tested on **completely unseen subjects**, which is why the accuracy dropped compared to Attempts 2-3.

**Result: 53.05% accuracy on 14,694 test samples from 4 unseen subjects**

**Outcome:** Lower raw accuracy than Attempts 2-3, but a much more honest evaluation. The model was now being tested the way it would be used in the real world.

---

## Attempt 5 — Final EEGNet Configuration (`final-model.ipynb`)

**Goal:** Improve subject-independent generalization through better preprocessing, loss functions, and training techniques.

This attempt involved the most experimentation. Multiple configurations were tested:

### Preprocessing Experiments
- **StandardScaler (SC)** — normalized each channel across all subjects
- **Euclidean Alignment (EA)** — a subject-independent preprocessing technique that re-centers each subject's data to a common space. Despite being theoretically sound, EA caused accuracy to drop from ~53% to ~30% (near random chance) and was abandoned

### Loss Function Experiments
- **CrossEntropy with class weights** — addressed the class imbalance between Rest (majority) and Left/Right (minority)
- **Focal Loss** — down-weighted easy examples to force the model to focus on harder Left/Right classifications. Improved minority class F1 but reduced overall accuracy

### Training Technique Experiments
- **DANN (Domain Adversarial Neural Network)** — added a subject classifier head with gradient reversal to force subject-invariant features. Despite theoretically addressing the core generalization problem, DANN consistently degraded performance — the best epoch was always before DANN activated at epoch 30
- **ReduceLROnPlateau scheduler** — repeatedly killed the learning rate too aggressively, causing training to stall
- **CosineAnnealing scheduler** — provided stable LR decay over 150 epochs without sudden drops

### Final Winning Configuration
After all experiments, the best configuration was the simplest:

| Component | Choice |
|-----------|--------|
| Preprocessing | StandardScaler |
| Loss Function | CrossEntropy with class weights [0.66, 1.30, 1.38] |
| Optimizer | Adam (lr=0.001, weight_decay=1e-3) |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-5) |
| Epochs | 150 |
| Batch Size | 128 |

**Final Result: 54.21% accuracy on 14,694 test samples from 4 completely unseen subjects**

---

## Phase 3 — Unity Integration

### Week 6 — Setting Up Inference Engine
The original plan called for Unity Sentis. In Unity 6, Sentis was rebranded to **Inference Engine 2.2.2** (`com.unity.ai.inference`). Installation required manually adding `"com.unity.sentis": "2.1.1"` to `manifest.json` since it did not appear in the Unity Registry UI.

### Week 7 — CSV Streamer
A C# script was written to read the test CSV file one epoch at a time, simulating a live EEG headset. The streamer passes each epoch to the inference model and updates a `latestPrediction` variable every 0.8 seconds.

Getting correct inference results in Unity required solving several non-obvious problems:

| Problem | Cause | Solution |
|---------|-------|----------|
| 30% accuracy in Unity vs 54% in Python | Model weights not loading (0MB) | Serialize `.onnx` to `.sentis` via Unity Inspector |
| Wrong predictions despite correct data | `GPUCompute` backend bug with depthwise convolutions | Switch to `BackendType.CPU` |
| Two-file ONNX export | PyTorch dynamo exporter splits weights | Use `onnx.save` with `save_as_external_data=False` |
| Wrong inference scores (inverted outputs) | BatchNorm folded incorrectly by dynamo exporter at opset 18 | Export with opset 15 — Unity Inference Engine supports opset 7-15 only |
| Asset import failed after file changes | Unity cached old model | Delete `Library` folder and reopen Unity |

### Week 8 — Real-Time Object Control
A `RotateObject.cs` script reads `latestPrediction` from the streamer every frame and rotates the cube accordingly:
- **Prediction 1 (Left):** rotate left
- **Prediction 2 (Right):** rotate right
- **Prediction 0 (Rest):** no movement

---

## Key Lessons Learned

1. **Subject-independent BCI is genuinely hard.** The 10%+ accuracy drop from same-subject to unseen-subject testing reflects a real challenge in the field. 54% on unseen subjects is within the range reported in academic literature for 3-class subject-independent EEG classification.

2. **Simpler often wins.** The most complex configurations (EA, Focal Loss, DANN) all underperformed the simple weighted CrossEntropy baseline. Adding complexity without a clear signal that the baseline is saturated is usually counterproductive.

3. **The learning rate scheduler matters more than the loss function.** `ReduceLROnPlateau` with aggressive decay was the hidden killer in multiple training runs. Switching to `CosineAnnealing` was more impactful than any loss function change.

4. **Unity's Inference Engine has non-obvious requirements.** The gap between Python accuracy and Unity accuracy was entirely due to how the model was loaded, not the model itself. The `.sentis` serialization step and `BackendType.CPU` requirement are not well documented.

5. **Test on the right distribution from the start.** Attempts 2-3 showed 66%+ accuracy but were tested on the same subjects as training. Switching to `GroupShuffleSplit` in Attempt 4 gave a more honest picture of real-world performance.
