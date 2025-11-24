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

def convert_keras_to_tflite():
    # ... (rest of the function is the same) ...
    if not os.path.exists(H5_MODEL_PATH):
        # Now gives a more helpful error message reflecting the correct path
        logging.error(f"Keras model not found at: {H5_MODEL_PATH}. Run auto_retrain.py first.")
        return

    try:
        logging.info(f"Loading Keras model from {H5_MODEL_PATH}...")
        model = load_model(H5_MODEL_PATH, compile=False)

        # ... (TFLite conversion logic) ...

        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        logging.info(f"✅ Conversion successful! Saved to {TFLITE_MODEL_PATH}")
        logging.info("You can now deploy this TFLite model to Android for real-time embeddings.")

    except Exception as e:
        logging.error(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
