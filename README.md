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
