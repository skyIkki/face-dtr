#!/usr/bin/env python3
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CORRECTED PATHS ---
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"

# Input model path must now point to the artifacts directory
H5_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.h5") 

# Output model path must also go into the artifacts directory
TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.tflite")

# In convert_to_tflite.py:
def convert_keras_to_tflite():
    # ... (os.path.exists check and model loading) ...
    
    try:
        logging.info("Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Optional: enable size/speed optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        logging.info("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # Save and log success INSIDE the try block
        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        logging.info(f"✅ Conversion successful! Saved to {TFLITE_MODEL_PATH}")
        logging.info("You can now deploy this TFLite model to Android for real-time embeddings.")

    except Exception as e:
        # If conversion fails, this catches it and logs it properly
        logging.error(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
