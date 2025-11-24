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
# In convert_to_tflite.py:

def convert_keras_to_tflite():
    # 1. Check for file existence (outside of the main try block)
    if not os.path.exists(H5_MODEL_PATH):
        logging.error(f"Keras model not found at: {H5_MODEL_PATH}. Run auto_retrain.py first.")
        return

    # 2. Try loading the model
    try:
        logging.info(f"Loading Keras model from {H5_MODEL_PATH}...")
        # Define 'model' here, making it accessible for the conversion logic
        model = load_model(H5_MODEL_PATH, compile=False) 
    except Exception as e:
        # If loading fails (e.g., corrupt file), log and exit early
        logging.error(f"Failed to load Keras model: {e}")
        return

    # 3. Perform the conversion (in a separate try block for cleaner error handling)
    try:
        logging.info("Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optional: enable size/speed optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        logging.info("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # Save the result
        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        logging.info(f"✅ Conversion successful! Saved to {TFLITE_MODEL_PATH}")
        logging.info("You can now deploy this TFLite model to Android for real-time embeddings.")

    except Exception as e:
        # Catch errors during the conversion/saving process
        logging.error(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
