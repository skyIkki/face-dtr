import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

H5_MODEL_PATH = "face_recognition_model.h5"
TFLITE_MODEL_PATH = "face_recognition_model.tflite"

def convert_keras_to_tflite():
    """
    Loads the Keras .h5 model and converts it to a TensorFlow Lite (.tflite) 
    format optimized for mobile devices.
    """
    if not os.path.exists(H5_MODEL_PATH):
        logging.error(f"Keras model not found at: {H5_MODEL_PATH}. Run auto_retrain.py first.")
        return

    try:
        # 1. Load the Keras model
        logging.info(f"Loading Keras model from {H5_MODEL_PATH}...")
        model = load_model(H5_MODEL_PATH)

        # 2. Initialize the TFLite Converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and speed (default optimization)
        # For better performance, you can explore optimization options like 
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 3. Perform the conversion
        logging.info("Converting model to TensorFlow Lite format...")
        tflite_model = converter.convert()

        # 4. Save the TFLite model
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
            
        logging.info(f"✅ Conversion successful! TFLite model saved to {TFLITE_MODEL_PATH}")
        
    except Exception as e:
        logging.error(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    convert_keras_to_tflite()
