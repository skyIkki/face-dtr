import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import firebase_admin
from firebase_admin import credentials, storage as fb_storage
import logging
import json
import base64

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_SAVE_PATH = "face_recognition_model.h5"
CLASS_MAPPING_FILE = "class_mapping.json"

# Image settings must match training settings
TARGET_SIZE = (160, 160)
SAMPLE_IMAGE_PATH = "sample_face_to_recognize.jpg" # Placeholder for image path
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Firebase Storage Configuration (Must match training script)
FIREBASE_BUCKET_NAME = "your-firebase-bucket.appspot.com"
# NOTE: The training script assumes FIREBASE_SERVICE_ACCOUNT_KEY env var is set.

# -----------------------------
# Firebase Setup and Download
# -----------------------------
def initialize_firebase():
    """Initializes Firebase Admin SDK."""
    if firebase_admin._apps:
        return

    logging.info("Initializing Firebase Admin SDK for inference...")
    try:
        service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
        if not service_account_base64:
            logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found. Cannot initialize Firebase.")
            raise ValueError("Firebase Service Account Key missing.")

        service_account_json_decoded = base64.b64decode(service_account_base64).decode('utf-8')
        cred = credentials.Certificate(json.loads(service_account_json_decoded))
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': FIREBASE_BUCKET_NAME
        })
        logging.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Firebase Admin SDK: {e}")
        raise

def download_artifacts():
    """Downloads the latest model and class mapping from Firebase Storage."""
    initialize_firebase()
    bucket = fb_storage.bucket()

    # Download Model
    model_blob = bucket.blob(MODEL_SAVE_PATH)
    model_blob.download_to_filename(MODEL_SAVE_PATH)
    logging.info(f"✅ Downloaded latest Model: {MODEL_SAVE_PATH}")

    # Download Mapping
    mapping_blob = bucket.blob(CLASS_MAPPING_FILE)
    mapping_blob.download_to_filename(CLASS_MAPPING_FILE)
    logging.info(f"✅ Downloaded Class Mapping: {CLASS_MAPPING_FILE}")

# -----------------------------
# Preprocessing and Recognition
# -----------------------------
def preprocess_face(face_img):
    """
    Crops the face, resizes it to the target size, and normalizes it.
    Input is a raw OpenCV image (numpy array).
    Output is a Keras-ready tensor (1, TARGET_SIZE, TARGET_SIZE, 3).
    """
    # Resize face image to the model's expected input size
    face_resized = cv2.resize(face_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Convert to Keras tensor format (add batch dimension, normalize)
    img_array = keras_image.img_to_array(face_resized)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize
    
    return img_array

def recognize_face(model, class_mapping, image_path):
    """
    Detects faces in an image, preprocesses the first face found, 
    and uses the model to predict the identity.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not load image at path: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        logging.error("Could not load Haar cascade classifier.")
        return

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        logging.warning("No face detected in the image.")
        return
        
    logging.info(f"Detected {len(faces)} faces. Processing the largest one.")
    
    # Select the largest detected face (optional refinement)
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    
    # Extract the face ROI (Region of Interest)
    face_img = img[y:y+h, x:x+w]
    
    # Preprocess the face for the model
    input_tensor = preprocess_face(face_img)
    
    # Run prediction
    predictions = model.predict(input_tensor, verbose=0)
    
    # Get the class ID with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Map the index back to the class name
    predicted_label = class_mapping.get(str(predicted_class_index), "UNKNOWN_CLASS")

    print("\n--- Recognition Result ---")
    print(f"Predicted Identity: {predicted_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("--------------------------\n")
    
    # Optional: Draw results on the image and save
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{predicted_label} ({confidence:.2f})", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
    output_path = "recognition_output.jpg"
    cv2.imwrite(output_path, img)
    logging.info(f"Visual output saved to {output_path}")

# -----------------------------
# Main Inference Flow
# -----------------------------
def main():
    try:
        # 1️⃣ Download latest model and mapping
        download_artifacts()
        
        # 2️⃣ Load model and mapping
        model = load_model(MODEL_SAVE_PATH)
        with open(CLASS_MAPPING_FILE, 'r') as f:
            class_mapping = json.load(f)
        
        logging.info(f"Model loaded. Total classes recognized: {len(class_mapping)}")
        
        # 3️⃣ Ensure a sample image exists (for demonstration purposes)
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            logging.warning(f"Could not find {SAMPLE_IMAGE_PATH}. Please provide a sample image.")
            return

        # 4️⃣ Perform face recognition
        recognize_face(model, class_mapping, SAMPLE_IMAGE_PATH)

    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
    finally:
        # Cleanup local artifacts after use
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
        if os.path.exists(CLASS_MAPPING_FILE):
            os.remove(CLASS_MAPPING_FILE)
        logging.info("Local model artifacts cleaned up.")

if __name__ == "__main__":
    main()
