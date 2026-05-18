# Pediatric Pneumonia Detection using Genetic Algorithm Feature Selection

A multimodal machine learning pipeline for detecting pneumonia and identifying its cause (Bacterial vs Viral) from chest X-rays combined with synthetic clinical data. A **Genetic Algorithm (GA)** is used for intelligent feature selection across the fused feature space.

---

## Project Overview

This project demonstrates:
- **Multimodal fusion**: CNN image features (MobileNetV2) + clinical tabular features
- **Genetic Algorithm** for automated feature selection
- **Binary classification**: Normal vs Pneumonia
- **Multi-class classification**: Normal vs Bacterial Pneumonia vs Viral Pneumonia

### Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle.

> **Note:** Clinical features (temperature, WBC count, SpO2, etc.) are synthetically generated and correlated with the image labels to simulate a real multimodal dataset. They do **not** come from real patients.

---

## Project Structure

```
Mini_Proj/
│
├── data/                        # Created after running download_data.py
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── test/
│       ├── val/
│       └── clinical_data.csv    # Auto-generated synthetic clinical data
│
├── outputs/                     # Created after running train_evaluate.py
│   ├── cm_pneumonia.png
│   └── cm_cause.png
│
├── data_loader.py               # Loads images + clinical data, preprocesses
├── download_data.py             # Downloads Kaggle dataset + generates clinical data
├── feature_extractor.py         # MobileNetV2 CNN feature extraction + fusion
├── genetic_algorithm.py         # GA-based feature selection
├── train_evaluate.py            # Main training & evaluation pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/pediatric-pneumonia-detection.git
cd pediatric-pneumonia-detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API
1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<user>\.kaggle\` (Windows)
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

## Running the Project

### Step 1 — Download Data
```bash
python download_data.py
```
This downloads the Kaggle chest X-ray dataset and generates `clinical_data.csv`.

### Step 2 — Run the Full Pipeline
```bash
python train_evaluate.py
```

For a quick test with a small sample:
```python
# In train_evaluate.py, change:
loader.load_data(max_samples=200)   # Use a small number
# GA settings:
GeneticAlgorithmFeatureSelection(population_size=10, generations=5)
```

For a full run, set `max_samples=None` and increase GA population/generations.

---

## Pipeline Architecture

```
Chest X-Ray Images
        │
        ▼
  MobileNetV2 (pretrained, ImageNet)
  GlobalAveragePooling2D
        │
        ▼                    Synthetic Clinical Features
  CNN Features (1280-d)  +   (temperature, WBC, SpO2, ...)
        │                              │
        └──────────── Fusion ──────────┘
                          │
                          ▼
              Genetic Algorithm Feature Selection
                          │
                          ▼
              RandomForest Classifier
               /                    \
    Binary Classification       Multi-class Classification
    (Normal vs Pneumonia)    (Normal vs Bacteria vs Virus)
```

---

## Results

Outputs are saved to the `outputs/` directory:
- `cm_pneumonia.png` — Confusion matrix for binary classification
- `cm_cause.png` — Confusion matrix for cause classification

> Results vary depending on `max_samples` and GA parameters. Use `max_samples=None` for best accuracy.

---

## Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| CNN Backbone | MobileNetV2 | Lightweight, pretrained on ImageNet |
| Image Size | 128×128 | Balance between speed and detail |
| Feature Selection | Genetic Algorithm | Handles mixed (image+clinical) feature spaces |
| Classifier | Random Forest | Robust, interpretable, handles high dimensions |
| Clinical Data | Synthetic | Demonstrates multimodal pipeline without real EHR data |

---

## Limitations

- Clinical features are **synthetic** and not clinically validated
- Small sample sizes significantly affect accuracy
- The GA is computationally expensive; increase generations for better results
- `Unknown` pneumonia cause samples (those without `bacteria`/`virus` in filename) are excluded from multi-class training

---

## Requirements

- Python 3.8–3.10
- TensorFlow < 2.16
- See `requirements.txt` for full list

---

## License

This project is for educational purposes. The chest X-ray dataset is subject to [Kaggle's terms of use](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
