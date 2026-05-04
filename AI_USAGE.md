# AI Usage Documentation

## Overview
This project uses machine learning techniques to classify EEG signals into cognitive states (drowsy, mid, stimulated) and simulate real-time responses (e.g., dimming lights). AI tools were used throughout development to assist with data processing, model implementation, and debugging.

---

## AI Tool Usage

### 1. EEG Data Pipeline & Preprocessing
**Tool Used:** ChatGPT  

**Prompt/Request:**  
“Help me build a full EEG preprocessing pipeline including loading data, organizing metadata, and creating LOSO splits.”

**What Was Generated:**
Code to:
- Load EEG `.csv` files from structured directories  
- Build a metadata DataFrame  
- Map raw labels to standardized classes  
- Implement Leave-One-Subject-Out (LOSO) cross-validation  

**Modifications we made:**
- Added filtering for invalid folders/files  
- Adjusted class mappings (Stimulated, Semi, rest)  
- Verified subject grouping manually  

**What we learned:**
- Importance of LOSO for avoiding subject leakage  
- How to structure datasets for cross-subject generalization  

---

### 2. Signal Processing & Feature Extraction
**Tool Used:** ChatGPT  

**Prompt/Request:**  
“Add bandpass filtering, windowing, and feature extraction for EEG signals including bandpower and ratios.”

**What Was Generated:**
- Bandpass filtering function (Butterworth filter)  
- Sliding window segmentation (4s windows, 50% overlap)  
- Feature extraction including:
  - Mean, variance, standard deviation  
  - Delta, theta, alpha, beta bandpower  
  - Relative bandpower  
  - Ratios (theta/alpha, theta/beta, alpha/beta)  

**Modifications we made:**
- Adjusted frequency ranges  
- Tuned window size and overlap  
- Added numerical stability (`eps`)  
- Integrated features into model pipeline  

**What we learned:**
- EEG features are highly sensitive to preprocessing  
- Ratio features often capture cognitive state better than raw power  

---

### 3. Model Tuning (SVM + Random Forest)
**Tool Used:** ChatGPT  

**Prompt/Request:**  
“Help me choose hyperparameters for my SVM models (Polynomial and RBF) and Random Forest”

**What Was Generated:**
Pipelines using:
- `StandardScaler`, `SVC` (poly and RBF kernels)  
- Random Forest (100 estimators, full depth)  

**Modifications we made:**
- Added per-subject normalization  
- Compared multiple models (Poly, RBF, RF)  
- Stored predictions across folds  

**What we learned:**
- Macro F1 is better than accuracy for imbalanced classes  
- RBF kernels often generalize better than polynomial in EEG tasks  

---

### 4. Visualization & Analysis
**Tool Used:** ChatGPT  

**Prompt/Request:**  
“Generate plots for PSD, bandpower distributions, and confusion matrices.”

**What Was Generated:**
- PSD plots using Welch’s method  
- Boxplots for bandpower features  
- Normalized PSD comparisons across subjects  
- Confusion matrix heatmaps  

**Modifications we made:**
- Standardized axes across subjects  
- Improved labeling and readability  

**What we learned:**
- Visualization is critical for understanding EEG differences  
- Subject variability is clearly visible in PSD plots  

---

### 5. Real-Time Classification Simulation
**Tool Used:** ChatGPT  

**Prompt/Request:**  
“Create a real-time classification script that maps predictions to light dimming.”

**What Was Generated:**
Script that:
- Loads trained model (`joblib`)  
- Simulates streaming EEG windows  
- Predicts cognitive state  
- Runs visualization of light changes based on predictions  

**Modifications we made:**
- Adjusted timing intervals  
- Integrated with saved test data  
- Customized output formatting  
- Added sustained window count logic  

**What we learned:**
- Real-time systems require consistent feature pipelines  
- Model and preprocessing must match exactly  

---

## Reflection

### How AI Helped Our Productivity
AI significantly accelerated development by:
- Generating full pipeline structures quickly  
- Helping debug complex issues (normalization, leakage)  
- Providing correct implementations of signal processing and ML methods  
- Allowing rapid experimentation with different models and techniques  

Without AI, implementing and debugging this entire pipeline would have taken substantially longer.

---

### What AI Was Not Helpful For
- Understanding why models performed poorly  
- Debugging subtle data leakage issues  
- Interpreting EEG signals at a deep conceptual level  
- Tuning hyperparameters effectively  

AI often provided correct code but required manual validation and conceptual understanding.

---

### How We Verified AI-Generated Code
We verified correctness by:
- Checking intermediate outputs (shapes, values, distributions)  
- Visualizing signals and features (PSD, bandpower plots)  
- Comparing model results across folds  
- Ensuring no data leakage (LOSO and time splits)  
- Cross-validating results with multiple models  

Additionally, we manually inspected:
- Feature extraction outputs  
- Confusion matrices  
- Per-subject performance differences  

---

## Summary
AI was used to assist in developing the overall architecture and improve implementation efficiency. All generated code was reviewed, modified, and validated to ensure correctness.
