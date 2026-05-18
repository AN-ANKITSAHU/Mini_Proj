import os
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Clinical feature columns used for model input
CLINICAL_FEATURE_COLS = [
    'temperature', 'wbc_count', 'oxygen_saturation',
    'cough_severity', 'chest_pain', 'fatigue',
    'noisy_feature_1', 'noisy_feature_2'
]

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "chest_xray")
        # FIX: clinical_data.csv is saved inside chest_xray/ by download_data.py
        self.clinical_file = os.path.join(data_dir, "chest_xray", "clinical_data.csv")

    def load_data(self, max_samples=None):
        """
        Loads the tabular clinical data and the corresponding chest X-ray images.

        Args:
            max_samples (int, optional): Limit number of samples for faster testing.

        Returns:
            images (np.ndarray): Preprocessed images of shape (N, 128, 128, 3).
            clinical_features_scaled (np.ndarray): Standardized clinical features (N, 8).
            y_binary (np.ndarray): Binary labels — 0: Normal, 1: Pneumonia.
            y_cause (np.ndarray): Multi-class labels — 0: Normal, 1: Bacteria, 2: Virus.
            scaler (StandardScaler): Fitted scaler (save this for inference).
        """
        if not os.path.exists(self.clinical_file):
            raise FileNotFoundError(
                f"Clinical data not found at {self.clinical_file}. "
                "Run 'python download_data.py' first."
            )

        df = pd.read_csv(self.clinical_file)

        # Shuffle with fixed seed for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if max_samples:
            df = df.head(max_samples)

        images = []
        valid_indices = []

        print("Loading images and preprocessing...")
        for idx, row in df.iterrows():
            img_path = os.path.join(self.img_dir, row['split'], row['label'], row['image_id'])

            if not os.path.exists(img_path):
                print(f"Warning: Image not found — {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img / 255.0  # Normalize pixel values to [0, 1]
                images.append(img)
                valid_indices.append(idx)

        if not images:
            raise RuntimeError("No valid images were loaded. Check your data directory structure.")

        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        images = np.array(images)

        # Extract and standardize clinical features
        clinical_features = df_valid[CLINICAL_FEATURE_COLS].values
        scaler = StandardScaler()
        clinical_features_scaled = scaler.fit_transform(clinical_features)

        # Binary target: 0 = Normal, 1 = Pneumonia
        y_binary = (df_valid['label'] == 'PNEUMONIA').astype(int).values

        # Multi-class target: 0 = Normal, 1 = Bacteria, 2 = Virus
        cause_map = {"Normal": 0, "Bacteria": 1, "Virus": 2}
        y_cause = df_valid['cause'].map(cause_map).fillna(-1).astype(int).values

        print(f"Loaded {len(images)} samples successfully.")
        return images, clinical_features_scaled, y_binary, y_cause, scaler


if __name__ == "__main__":
    loader = DataLoader()
    try:
        imgs, clinics, y_bin, y_cause, scaler = loader.load_data(max_samples=100)
        print(f"Images shape:   {imgs.shape}")
        print(f"Clinical shape: {clinics.shape}")
        print(f"Binary labels:  {np.bincount(y_bin)}")
    except Exception as e:
        print(f"Error loading data: {e}")
