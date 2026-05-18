import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input

class FeatureExtractor:
    def __init__(self, input_shape=(128, 128, 3)):
        print("Initializing Feature Extractor (MobileNetV2)...")
        # Base model without top classification layers
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Add GlobalAveragePooling to get a 1D vector per image
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        self.model = Model(inputs=base_model.input, outputs=x)
        
    def extract_image_features(self, images):
        """
        Takes a batch of preprocessed images (N, 128, 128, 3) 
        and returns advanced visual features.
        """
        print(f"Extracting CNN features for {len(images)} images...")
        features = self.model.predict(images, batch_size=32, verbose=1)
        return features
        
    def fuse_features(self, image_features, clinical_features):
        """
        Concatenates image features and clinical features into a single array
        """
        print(f"Fusing {image_features.shape[1]} image features with {clinical_features.shape[1]} clinical features.")
        fused = np.hstack((image_features, clinical_features))
        return fused

if __name__ == "__main__":
    # Dummy test
    dummy_imgs = np.random.rand(10, 128, 128, 3)
    dummy_clinical = np.random.rand(10, 8)
    
    extractor = FeatureExtractor()
    img_feats = extractor.extract_image_features(dummy_imgs)
    combined = extractor.fuse_features(img_feats, dummy_clinical)
    
    print(f"Combined features shape: {combined.shape}")
