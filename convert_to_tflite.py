#!/usr/bin/env python3
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
H5_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.h5")
TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.tflite")

def convert_keras_to_tflite():
    if not os.path.exists(H5_MODEL_PATH):
        logging.error(f"Keras model not found at {H5_MODEL_PATH}. Run auto_retrain.py first.")
        return

    try:
        logging.info(f"Loading Keras model from {H5_MODEL_PATH}...")
        model = load_model(H5_MODEL_PATH, compile=False)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    try:
        logging.info("Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Ensure input/output is float32 with [-1,1] normalization
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        logging.info("Converting model to TFLite...")
        tflite_model = converter.convert()

        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        logging.info(f"✅ Conversion successful! Saved to {TFLITE_MODEL_PATH}")
    except Exception as e:
        logging.error(f"TFLite conversion failed: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
