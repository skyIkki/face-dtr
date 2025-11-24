#!/usr/bin/env python3
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

H5_MODEL_PATH = "face_embedding_model.h5"
TFLITE_MODEL_PATH = "face_embedding_model.tflite"

def convert_keras_to_tflite():
    if not os.path.exists(H5_MODEL_PATH):
        logging.error(f"Keras model not found at: {H5_MODEL_PATH}. Run auto_retrain.py first.")
        return

    try:
        logging.info(f"Loading Keras model from {H5_MODEL_PATH}...")
        model = load_model(H5_MODEL_PATH, compile=False)  # embedding model does not need compilation

        logging.info("Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optional: enable size/speed optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Optional: dynamic range quantization
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

        logging.info("Converting model to TFLite format...")
        tflite_model = converter.convert()

        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        logging.info(f"✅ Conversion successful! Saved to {TFLITE_MODEL_PATH}")
        logging.info("You can now deploy this TFLite model to Android for real-time embeddings.")

    except Exception as e:
        logging.error(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
