import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader, CLINICAL_FEATURE_COLS
from feature_extractor import FeatureExtractor
from genetic_algorithm import GeneticAlgorithmFeatureSelection

OUTPUT_DIR = "outputs"


def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=== Pediatric Pneumonia Prediction using Genetic Algorithm ===\n")

    # 1. Load Data
    # Set max_samples=None for a full run; use a small number for quick testing.
    loader = DataLoader(data_dir="data")
    try:
        images, clinical_features_scaled, y_binary, y_cause, scaler = loader.load_data(max_samples=200)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        print("Please run 'python download_data.py' first to prepare the dataset.")
        return

    # 2. Extract CNN Image Features via MobileNetV2
    extractor = FeatureExtractor()
    image_features = extractor.extract_image_features(images)

    X_full = extractor.fuse_features(image_features, clinical_features_scaled)

    num_cnn = image_features.shape[1]
    num_clinical = clinical_features_scaled.shape[1]
    print(f"\nTotal Features: {X_full.shape[1]}  (CNN: {num_cnn}, Clinical: {num_clinical})\n")

    # FIX: Use only the 8 clinical feature names (not all CSV columns)
    cnn_feature_names = [f"cnn_feature_{i}" for i in range(num_cnn)]
    all_feature_names = np.array(cnn_feature_names + CLINICAL_FEATURE_COLS)

    # 3. Train/Test Split
    X_train, X_test, y_train_bin, y_test_bin, y_train_cause, y_test_cause = train_test_split(
        X_full, y_binary, y_cause,
        test_size=0.2, random_state=42, stratify=y_binary
    )

    # 4. Genetic Algorithm Feature Selection (optimizing for pneumonia detection)
    print("--- Running Genetic Algorithm for Feature Selection ---")
    ga = GeneticAlgorithmFeatureSelection(population_size=10, generations=5, mutation_rate=0.1)
    best_chromosome, best_fitness = ga.optimize(X_train, y_train_bin)

    selected_indices = np.where(best_chromosome == 1)[0]
    selected_feature_names = all_feature_names[selected_indices]

    print("\n=== GA Optimization Results ===")
    print(f"Best Validation Accuracy (GA):  {best_fitness:.4f}")
    print(f"Features Selected:              {len(selected_indices)} / {X_full.shape[1]}")

    selected_clinical = [f for f in selected_feature_names if f in CLINICAL_FEATURE_COLS]
    print(f"Selected Clinical Features:     {selected_clinical}")

    # 5. Prepare selected feature subsets
    X_train_sel = X_train[:, selected_indices]
    X_test_sel  = X_test[:, selected_indices]

    # --- Model 1: Binary — Normal vs Pneumonia ---
    print("\n--- Training Binary Model: Normal vs Pneumonia ---")
    clf_bin = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_bin.fit(X_train_sel, y_train_bin)

    y_pred_bin = clf_bin.predict(X_test_sel)
    print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
    print(classification_report(y_test_bin, y_pred_bin, target_names=["Normal", "Pneumonia"]))
    plot_confusion_matrix(y_test_bin, y_pred_bin, ["Normal", "Pneumonia"],
                          "Pneumonia Detection", "cm_pneumonia.png")

    # --- Model 2: Multi-class — Normal vs Bacteria vs Virus ---
    print("\n--- Training Multi-class Model: Normal vs Bacteria vs Virus ---")
    # Filter out any unknown-cause samples (label == -1)
    valid_mask = y_train_cause != -1
    clf_cause = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_cause.fit(X_train_sel[valid_mask], y_train_cause[valid_mask])

    valid_test_mask = y_test_cause != -1
    y_pred_cause = clf_cause.predict(X_test_sel[valid_test_mask])
    print(f"Accuracy: {accuracy_score(y_test_cause[valid_test_mask], y_pred_cause):.4f}")
    print(classification_report(y_test_cause[valid_test_mask], y_pred_cause,
                                target_names=["Normal", "Bacteria", "Virus"]))
    plot_confusion_matrix(y_test_cause[valid_test_mask], y_pred_cause,
                          ["Normal", "Bacteria", "Virus"],
                          "Cause Prediction", "cm_cause.png")

    print(f"\nPipeline complete. Results saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
