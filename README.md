# GASP: Genetic Algorithm-Optimized Hybrid SVM for Pediatric Pneumonia Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **GASP** stands for **G**enetic **A**lgorithm-Optimized Hybrid **S**VM for Sustainable and Accessible **P**ediatric Pneumonia Screening in Resource-Constrained Settings.

![Accuracy](https://img.shields.io/badge/Accuracy-94.80%25-success)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9815-success)
![Pneumonia Recall](https://img.shields.io/badge/Recall-98%25-success)

---

## 📋 Overview

Pneumonia is one of the leading causes of mortality in children under five years old, claiming approximately **700,000 lives annually** worldwide. Early and accurate detection from chest X-ray images can enable timely clinical intervention and save lives.

**GASP** is an efficient, **CPU-optimized** automated diagnostic framework designed specifically for pediatric pneumonia detection. Unlike deep learning approaches that require expensive GPU infrastructure, GASP achieves state-of-the-art performance using classical machine learning techniques optimized through evolutionary algorithms.

### Key Highlights

- **94.80%** test accuracy
- **98%** pneumonia recall (sensitivity)
- **0.9815** ROC-AUC score
- **< 1 second** per prediction on standard CPU hardware
- Designed for **resource-constrained clinical environments**

---

## 🎯 Problem Statement

Nearly **99% of pneumonia mortalities occur in Low and Middle-Income Countries (LMICs)** where:
- Radiologist access is severely limited
- GPU-enabled computing infrastructure is scarce
- Cost-effective diagnostic solutions are critical

Traditional deep learning models require substantial computational resources, creating barriers to deployment in these settings. GASP addresses this gap by providing a lightweight yet accurate diagnostic tool that operates efficiently on standard computing hardware.

---

## 🏗️ Architecture

The framework follows a **five-layer pipeline**:

```
┌─────────────────────────────────────────────────────────┐
│  Layer I: Data Acquisition & Preprocessing              │
│  ├─ Grayscale conversion                                │
│  ├─ 64×64 pixel resizing                                │
│  └─ Normalization                                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer II: HOG+LBP Feature Extraction                   │
│  ├─ HOG: 1,764 features (edges & structure)            │
│  ├─ LBP: 10 features (texture patterns)                │
│  └─ Combined: 1,774-dimensional feature vector         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer III: Leakage-Free Data Partitioning              │
│  └─ 65% Train / 15% Validation / 20% Test              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer IV: Genetic Algorithm Hyperparameter Tuning      │
│  ├─ Population: 20 chromosomes                          │
│  ├─ Generations: 15                                     │
│  ├─ Optimizes: C, γ, kernel type                       │
│  └─ Max 300 SVM calls (vs thousands in grid search)    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer V: Final SVM Training & Evaluation               │
│  ├─ Optimal hyperparameters: C=3406.35, γ=0.000994     │
│  ├─ RBF kernel                                          │
│  └─ Class-weighted for imbalanced data (1:2.7 ratio)   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### Feature Engineering

#### **Histogram of Oriented Gradients (HOG)**
- **Purpose**: Captures lung field boundaries, consolidation edges, and opacity gradients
- **Configuration**: 
  - 9 orientations
  - 8×8 pixels per cell
  - 2×2 cells per block
- **Output**: 1,764 features

#### **Local Binary Patterns (LBP)**
- **Purpose**: Captures alveolar texture and parenchymal density changes
- **Configuration**:
  - P=8 neighbors
  - R=1 radius
  - Uniform mapping with 10-bin histogram
- **Output**: 10 features

### Genetic Algorithm Optimization

| Parameter | Value |
|-----------|-------|
| Population size | 20 |
| Generations | 15 |
| Crossover probability | 0.60 |
| Mutation probability | 0.30 |
| Selection method | Tournament (size=3) |
| Crossover operator | Blend crossover (α=0.3) |
| Fitness function | Validation accuracy |

**Optimized Hyperparameters**:
- **C** = 3406.35
- **γ** = 0.000994
- **Kernel** = RBF (Radial Basis Function)

---

## 📊 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Test Accuracy | **94.80%** |
| Pneumonia Recall (Sensitivity) | **98%** |
| ROC-AUC | **0.9815** |
| Inference Time | **< 1 second/image** |

### Why This Matters

- **High Recall (98%)**: Critical for medical screening — minimizes false negatives, ensuring pneumonia cases are rarely missed
- **Strong ROC-AUC (0.9815)**: Excellent discrimination between pneumonia and normal cases
- **CPU-Optimized**: Runs on standard hardware, making it deployable in resource-limited settings
- **Fast Inference**: Real-time diagnosis capability for clinical workflows

---

## 🗂️ Dataset

**Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Provider**: Guangzhou Women and Children's Medical Center

**Composition**:
- **5,856** pediatric chest X-ray images
- **Age Group**: 1-5 years (12-60 months)
- **Classes**: NORMAL vs PNEUMONIA
- **Class Ratio**: 1:2.7 (imbalanced, handled via class weighting)

**Split Strategy**:
- Training: 65%
- Validation: 15%
- Testing: 20%

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Ayush-appie/mini.git
cd mini

# Install dependencies
pip install numpy pandas scikit-learn opencv-python matplotlib
pip install deap scikit-image tensorflow

# For Jupyter notebook support
pip install jupyter notebook
```

---

## 🚀 Usage

### Running the Main Pipeline

```python
# Open the main notebook
jupyter notebook pneumonia_ga_project\ \(1\).ipynb
```

### Quick Start Example

```python
import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern

# Load and preprocess image
img = cv2.imread('chest_xray.jpg', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (64, 64))

# Extract HOG features
hog_features = hog(img_resized, orientations=9, 
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2))

# Extract LBP features
lbp = local_binary_pattern(img_resized, P=8, R=1, method='uniform')
lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))

# Combine features
features = np.concatenate([hog_features, lbp_hist])

# Load trained model and predict
# model = joblib.load('gasp_model.pkl')
# prediction = model.predict([features])
```

---

## 📁 Repository Structure

```
mini/
├── pneumonia_ga_project (1).ipynb    # Main implementation notebook
├── gasp (3) (1).pdf                  # Research paper
├── KIIT_Report (2).pdf               # Project report
├── Untitled16.ipynb                  # Experimentation notebook
├── Untitled19.ipynb                  # Additional experiments
├── Untitled8.ipynb                   # Feature engineering tests
└── README.md                         # This file
```

---

## 🎓 Research Paper

This project is based on the research paper:

**"GASP: Genetic Algorithm-Optimized Hybrid SVM for Sustainable and Accessible Pediatric Pneumonia Screening in Resource-Constrained Settings"**

**Authors**: Ayush Parashar, Ankit Sahu, Aryan Mishra, Tanisha Mallick, Tannistha Ash, Ranjita Kumari Dash

**Institution**: Department of Computer Science & Engineering, KIIT University, Bhubaneswar, India

---

## 🔍 Why GASP Over Deep Learning?

| Aspect | GASP | Deep Learning CNNs |
|--------|------|-------------------|
| **Hardware Requirements** | Standard CPU | GPU required |
| **Training Time** | Minutes | Hours to days |
| **Inference Speed** | < 1 second | 1-2 seconds |
| **Model Size** | ~10 MB | 100+ MB |
| **Interpretability** | High (feature-based) | Low (black box) |
| **Deployment Cost** | Low | High |
| **Suitability for LMICs** | ✅ Excellent | ❌ Limited |

---

## 🌍 Real-World Impact

GASP is designed specifically for deployment in:

- Rural healthcare centers
- Mobile diagnostic units
- Community health programs
- Low and Middle-Income Countries (LMICs)
- Settings with limited radiologist availability

By providing accurate, fast, and resource-efficient pneumonia screening, GASP can help:
- Reduce child mortality rates
- Enable early intervention
- Support clinical decision-making
- Bridge the healthcare gap in underserved regions

---

## 🔬 Technical Innovations

1. **Hybrid Feature Engineering**: Combines structural (HOG) and textural (LBP) information specifically optimized for pediatric chest X-rays

2. **Evolutionary Hyperparameter Optimization**: GA-based search explores parameter space more efficiently than grid search (300 vs 10,000+ evaluations)

3. **Leakage-Free Pipeline**: Strict data partitioning prevents information leakage and ensures fair evaluation

4. **Class Imbalance Handling**: Weighted SVM with `class_weight='balanced'` addresses the 1:2.7 class distribution

---

## 📈 Future Work

- [ ] Extend to multi-class classification (bacterial vs viral pneumonia)
- [ ] Integration with PACS systems for clinical deployment
- [ ] Mobile application development for field diagnostics
- [ ] Cross-dataset validation on other pediatric X-ray datasets
- [ ] Explainable AI features (SHAP values, attention maps)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Ayush Parashar**
- Email: ayush.parashar995@gmail.com
- GitHub: [@Ayush-appie](https://github.com/Ayush-appie)
- Institution: KIIT University, Bhubaneswar

---

## 🙏 Acknowledgments

- Guangzhou Women and Children's Medical Center for the dataset
- Kaggle for hosting the Chest X-Ray Images (Pneumonia) dataset
- Dr. Ranjita Kumari Dash (Project Advisor)
- KIIT School of Computer Engineering
- Research team members: Ankit Sahu, Aryan Mishra, Tanisha Mallick, Tannistha Ash

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{parashar2024gasp,
  title={GASP: Genetic Algorithm-Optimized Hybrid SVM for Sustainable and Accessible Pediatric Pneumonia Screening in Resource-Constrained Settings},
  author={Parashar, Ayush and Sahu, Ankit and Mishra, Aryan and Mallick, Tanisha and Ash, Tannistha and Dash, Ranjita Kumari},
  journal={Department of Computer Science \& Engineering, KIIT University},
  year={2024}
}
```

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ for global child health**

*Empowering healthcare in resource-constrained settings*

</div>
