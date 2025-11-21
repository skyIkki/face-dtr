import os
import json
from firebase_admin import initialize_app, storage, credentials
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TFLITE_MODEL_PATH = "face_recognition_model.tflite"
CLASS_MAPPING_PATH = "class_mapping.json"
FIREBASE_BUCKET_NAME = "your-firebase-bucket.appspot.com" # CHANGE THIS TO YOUR ACTUAL BUCKET NAME

def deploy_tflite_artifacts():
    """
    Deploys the optimized TFLite model and the class mapping to Firebase Storage 
    for consumption by mobile applications.
    """
    if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(CLASS_MAPPING_PATH):
        logging.error("Required artifacts not found. Did the training and conversion steps run successfully?")
        return

    try:
        # 1. Initialize Firebase Admin SDK using the secret key
        service_account_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
        if not service_account_key:
            logging.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set.")
            return

        # Write the service account JSON string to a temporary file
        temp_creds_path = "firebase_creds.json"
        with open(temp_creds_path, "w") as f:
            f.write(service_account_key)

        cred = credentials.Certificate(temp_creds_path)
        app = initialize_app(cred, {'storageBucket': FIREBASE_BUCKET_NAME})
        bucket = storage.bucket(app=app)
        
        # 2. Deploy the TFLite Model
        tflite_blob = bucket.blob(f"mobile_artifacts/{TFLITE_MODEL_PATH}")
        tflite_blob.upload_from_filename(TFLITE_MODEL_PATH)
        logging.info(f"✅ Successfully deployed TFLite model to gs://{FIREBASE_BUCKET_NAME}/mobile_artifacts/{TFLITE_MODEL_PATH}")

        # 3. Deploy the Class Mapping
        mapping_blob = bucket.blob(f"mobile_artifacts/{CLASS_MAPPING_PATH}")
        mapping_blob.upload_from_filename(CLASS_MAPPING_PATH)
        logging.info(f"✅ Successfully deployed Class Mapping to gs://{FIREBASE_BUCKET_NAME}/mobile_artifacts/{CLASS_MAPPING_PATH}")
        
        # 4. Clean up temporary credentials file
        os.remove(temp_creds_path)

    except Exception as e:
        logging.error(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy_tflite_artifacts()
