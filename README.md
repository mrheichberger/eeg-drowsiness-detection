# EEG Drowsiness Detection
## Project Overview
This project explores the use of EEG-based signal detection to classify between three states: Stimulated, semi-stimulated, and drowsy. The implementation of this classification is proposed to be used for adaptive environment control, specifically among physically limited people. The system processes EEG data, extracts features, and uses trained models to simulate real-time classification. This also has further applications in performance optimization and has the potential to be expanded to other household appliances.

## Dataset
The data recorded in this project was obtained from 4 participants over 45-minute sessions. The sessions were divided as follows:

15-minutes stimulated → Subway Surfers

15-minutes semi-stimulated → PVT

15-minutes “drowsy” → Dot focus


## Methodology
The preprocessing pipeline started with truncation of datasets to ensure equal numbers of samples per participant. A bandpass filter was used to remove noise and retain relevant EEG information. Additionally, to increase the number of samples, a sliding window was used with a window size of 4 seconds and 50% overlap.

## Feature Extraction
The extracted features used in this project are:

Statistical measurements → Mean, std, variance

EEG-related frequency bands → alpha, beta, theta

  - Individual bands
  - Ratios
  - Relative powers

## Models
This project implemented SVM with two kernel types and also tried a random forest model:

- RBF SVM
- Polynomial SVM
- Random Forest

Models evaluated using:
- Accuracy
- F1 scores
- Confusion matrices


## Split Strategy
Two methods were implemented to split the data obtained in this project.

- LOSO 
- 80/20 Time-based Split

The time based split introduces data leakage; however, that was slightly mitigated with a 2-second purge gap. The accuracy obtained with the temporal split was likely still inflated due to high correlation of nearby samples, but was included for a more demonstrative real-time implementation.

## Real-time Simulation
An WebSocket server streams predictions to an HTML interface which:
Displays true and predicted classifications
Has varying brightness levels corresponding to classifications
Uses a sustained window count to mimic a more practical implementation

## How to Run
1.) Install libraries

2.) Run the realtime_server.py file

3.) Open the UI → Real_time_light.html






