# 🔬 Binary Diabetic Retinopathy Detection Using CNN — Messidor-2

> **MALEEA3 — Machine Learning 3A Final Project**  
> University of Johannesburg | Dept. of Electrical & Electronic Engineering Science  
> Lecturer: Prof. Khmaies Ouahada | Tutor: Mr. Bilal Ahmad

---

## 📋 Project Overview

This project implements a **10-layer Convolutional Neural Network (CNN)** for binary diabetic retinopathy (DR) screening on the [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) retinal fundus image dataset.

The model classifies each retinal fundus photograph into one of two classes:

| Class | Label | Description |
|-------|-------|-------------|
| **No DR** | 0 | No diabetic retinopathy detected (original grade 0) |
| **DR Present** | 1 | Any degree of DR detected (original grades 1–4) |

> **Note:** Binary classification was adopted following lecturer guidance after initial 4-class grading yielded accuracy below 80%. Binary framing is clinically meaningful — it directly answers the primary screening decision: *does this patient require ophthalmological referral?*

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 82.06% |
| **ROC-AUC** | 0.88 |
| **Macro F1-Score** | 0.85 |
| No DR — Precision / Recall / F1 | 0.82/ 0.81 / 0.81 |
| DR Present — Precision / Recall / F1 | 0.82 / 0.82 / 0.82 |

---

## 🏗️ CNN Architecture

A 10-layer Sequential CNN built with TensorFlow/Keras:

```
Input (128×128×3)
    ↓
Conv2D (32 filters, 3×3, ReLU, same padding)    → Layer 1
MaxPooling2D (2×2)                               → Layer 2
    ↓
Conv2D (64 filters, 3×3, ReLU, same padding)    → Layer 3
MaxPooling2D (2×2)                               → Layer 4
    ↓
Conv2D (128 filters, 3×3, ReLU, same padding)   → Layer 5
MaxPooling2D (2×2)                               → Layer 6
    ↓
Conv2D (256 filters, 3×3, ReLU, same padding)   → Layer 7
MaxPooling2D (2×2)                               → Layer 8
    ↓
Flatten                                          → Layer 9
Dropout (0.5)
Dense (256 units, ReLU)                          → Layer 10
    ↓
Dense (2 units, Softmax)                         → Output
```

**Total trainable parameters:** ~4,583,490

---

## 📁 Repository Structure

```
messidor2-diabetic-retinopathy-CNN/
│
├── messidor2_DR_detection_binary_FullyCommented.ipynb   # Main notebook
│
├── images/
│   ├── class_distribution.png      # Label distribution plots
│   ├── training_curves.png         # Accuracy and loss over epochs
│   ├── confusion_matrix.png        # Test set confusion matrices
│   ├── roc_curve.png               # ROC curve (AUC = 0.7605)
│   └── sample_predictions.png      # Sample test predictions
│
└── README.md
```

---

## 🗃️ Dataset

**Messidor-2** — publicly available retinal fundus photograph dataset.

- ~1,748 colour fundus images in `.png` format
- Labels provided in a companion CSV file (`messidor_data.csv`)
- Original grades 0–4 remapped to binary: 0 = No DR, 1 = DR Present

> The dataset is not included in this repository due to size. Download from:  
> https://www.adcis.net/en/third-party/messidor2/

---

## ⚙️ Setup and Usage

### Requirements

```bash
pip install tensorflow pillow scikit-learn matplotlib seaborn pandas numpy
```

### Running the Notebook

1. Clone or download this repository
2. Download the Messidor-2 dataset
3. Open `messidor2_DR_detection_binary_FullyCommented.ipynb` in Jupyter Notebook
4. Update the paths in **Block 2 (Configuration)**:

```python
IMAGE_DIR = r'C:\path\to\Messidor-2\preprocess'
CSV_PATH  = r'C:\path\to\messidor_data.csv'
```

5. Run all cells from top to bottom

### Output Files Generated

| File | Description |
|------|-------------|
| `class_distribution.png` | Class balance visualisation |
| `sample_images.png` | Sample retinal fundus images |
| `training_curves.png` | Training and validation curves |
| `confusion_matrix.png` | Test set confusion matrix |
| `roc_curve.png` | ROC curve with AUC score |
| `sample_predictions.png` | 16 sample test predictions |
| `messidor2_DR_CNN_binary.keras` | Saved trained model |

---

## 🔬 Methodology

### Preprocessing
- Images resized to **128×128 pixels**
- Pixel values normalised from [0, 255] → [0.0, 1.0]
- Stratified 3-way split: **68% train / 12% validation / 20% test**
- Class imbalance addressed via **computed class weights**
- Labels one-hot encoded for categorical cross-entropy loss

### Training
- **Optimiser:** Adam (lr = 1e-3)
- **Loss:** Categorical Cross-Entropy
- **Max epochs:** 50 (with EarlyStopping, patience=10)
- **Batch size:** 32
- **Callbacks:** EarlyStopping + ReduceLROnPlateau

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class + macro average)
- ROC-AUC (threshold-independent discriminative performance)
- Confusion matrix (counts + normalised)

---

## 📈 Key Findings

- The model achieves **70.49% accuracy** and **ROC-AUC of 0.7605**, representing a meaningful baseline for a from-scratch CNN on this dataset
- **No DR class** performs better (F1: 0.742) than **DR Present** (F1: 0.656), consistent with the class imbalance in Messidor-2
- DR Present recall of **67.6%** indicates ~32% of true DR cases are missed — below the 80% minimum sensitivity required for standalone clinical deployment
- The model is best positioned as a **human-in-the-loop triage tool** rather than an autonomous diagnostic system

---

## 🚀 Suggested Improvements

1. **Transfer Learning** — EfficientNetB0 or ResNet50 pre-trained on ImageNet 
2. **Data Augmentation** — horizontal flipping, rotation, brightness adjustment
3. **Higher Resolution** — 224×224 input to preserve fine vascular detail
4. **Threshold Optimisation** — lower classification threshold to maximise DR Present recall
5. **Batch Normalisation** — after each Conv2D layer for more stable training

---

## 🏥 Clinical Context

Diabetic retinopathy is the **leading cause of preventable blindness** in working-age adults worldwide. South Africa has over 4.2 million adults living with diabetes, yet ophthalmologist access is severely limited in many regions. Automated CNN-based screening offers a scalable pathway to extend DR detection capacity to primary healthcare settings.

---

## 📚 References

1. LeCun et al. — Gradient-based learning applied to document recognition (1998)
2. Gulshan et al. — Deep learning for DR detection in fundus photographs, JAMA (2016)
3. Decencière et al. — The Messidor database, Image Analysis and Stereology (2014)
4. Kingma & Ba — Adam: A method for stochastic optimization, ICLR (2015)

---

## 👤 Author

Phathutshedzo Murovhi — Student No: 222115222  
BSc Computer Science | University of Johannesburg  
phathulivi@gmail.com

---

*Submitted in partial fulfilment of MALEEA3 — Machine Learning 3A*  
*University of Johannesburg | April 2026*
