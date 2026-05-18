import os
import zipfile
import subprocess
import pandas as pd
import numpy as np

def download_kaggle_dataset():
    dataset = "paultimothymooney/chest-xray-pneumonia"
    download_dir = "data"
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"Downloading {dataset}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir], check=True)
        print("Download complete. Unzipping...")
        
        zip_path = os.path.join(download_dir, "chest-xray-pneumonia.zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
            
        print("Unzipping complete.")
        os.remove(zip_path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have kaggle installed and your kaggle.json is present in ~/.kaggle/")

def generate_synthetic_clinical_data(data_dir="data/chest_xray"):
    """
    Since the Kaggle dataset is images only, we generate synthetic clinical data 
    (symptoms, vitals) that strongly correlate with the true condition to demonstrate 
    the Genetic Algorithm's ability to select multimodal features.
    """
    splits = ["train", "test", "val"]
    
    all_records = []
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for category in ["NORMAL", "PNEUMONIA"]:
            cat_dir = os.path.join(split_dir, category)
            if not os.path.exists(cat_dir):
                continue
                
            for img_file in os.listdir(cat_dir):
                if not img_file.endswith((".jpeg", ".jpg", ".png")):
                    continue
                
                # Determine cause from filename or category
                cause = "Normal"
                if category == "PNEUMONIA":
                    if "bacteria" in img_file.lower():
                        cause = "Bacteria"
                    elif "virus" in img_file.lower():
                        cause = "Virus"
                    else:
                        cause = "Unknown"
                        
                # Generate correlated symptoms
                # 0-1 scale or actual values
                tmp = np.random.normal(37.0, 0.2)  # Normal temp
                wbc = np.random.normal(7000, 1000) # Normal WBC
                spo2 = np.random.normal(98, 1.0)   # Normal SpO2
                cough = np.random.randint(0, 3)    # 0=None, 1=Mild, 2=Severe
                chest_pain = np.random.randint(0, 2)
                fatigue = np.random.randint(0, 2)
                
                if cause == "Bacteria":
                    tmp = np.random.normal(39.5, 0.5)
                    wbc = np.random.normal(15000, 2000)
                    spo2 = np.random.normal(92, 2.0)
                    cough = np.random.randint(1, 4)
                    chest_pain = np.random.randint(1, 4)
                    fatigue = np.random.randint(2, 4)
                elif cause == "Virus":
                    tmp = np.random.normal(38.2, 0.4)
                    wbc = np.random.normal(8000, 1500)
                    spo2 = np.random.normal(94, 1.5)
                    cough = np.random.randint(1, 4)
                    chest_pain = np.random.randint(0, 3)
                    fatigue = np.random.randint(1, 4)
                
                # Add some noisy/irrelevant features for the GA to filter out
                noisy_feat_1 = np.random.random()
                noisy_feat_2 = np.random.random()
                
                all_records.append({
                    "image_id": img_file,
                    "split": split,
                    "label": category,
                    "cause": cause,
                    "temperature": tmp,
                    "wbc_count": wbc,
                    "oxygen_saturation": spo2,
                    "cough_severity": cough,
                    "chest_pain": chest_pain,
                    "fatigue": fatigue,
                    "noisy_feature_1": noisy_feat_1,
                    "noisy_feature_2": noisy_feat_2
                })
                
    df = pd.DataFrame(all_records)
    output_path = os.path.join(data_dir, "clinical_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic clinical data at {output_path} with {len(df)} records.")

if __name__ == "__main__":
    download_kaggle_dataset()
    if os.path.exists("data/chest_xray"):
        generate_synthetic_clinical_data()
    else:
        print("Cannot generate clinical data; image directory not found.")
