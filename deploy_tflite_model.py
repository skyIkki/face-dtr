import os
import json
from firebase_admin import initialize_app, storage, credentials
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TFLITE_MODEL_PATH = "face_recognition_model.tflite"
CLASS_MAPPING_PATH = "class_mapping.json"
# IMPORTANT: Replace with your actual project ID to ensure compatibility.
# The bucket name format is usually YOUR_PROJECT_ID.appspot.com
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.appspot.com" # Updated best practice for bucket naming

def deploy_tflite_artifacts():
    """
    Deploys the optimized TFLite model and the class mapping to Firebase Storage 
    for consumption by mobile applications.
    """
    if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(CLASS_MAPPING_PATH):
        logging.error("Required artifacts not found. Did the training and conversion steps run successfully?")
        return

    temp_creds_path = "firebase_creds.json"

    try:
        service_account_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
        if not service_account_key:
            logging.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set.")
            return

        # Explicitly check if the secret content is valid JSON.
        # This is where the original 'Expecting value' error occurs implicitly.
        try:
            logging.info("Attempting to parse service account key...")
            service_account_data = json.loads(service_account_key)
        except json.JSONDecodeError as jde:
            logging.error(f"Failed to decode FIREBASE_SERVICE_ACCOUNT_KEY as JSON. Error: {jde}")
            logging.error("Please ensure the secret in GitHub is a single, correctly formatted JSON string.")
            return

        # Write the guaranteed valid JSON structure to a temporary file
        with open(temp_creds_path, "w") as f:
            json.dump(service_account_data, f, indent=4)

        # 1. Initialize Firebase Admin SDK
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
        
    except Exception as e:
        logging.error(f"Deployment failed unexpectedly: {e}")
        
    finally:
        # 4. Clean up temporary credentials file (ensured to run even on error)
        if os.path.exists(temp_creds_path):
            os.remove(temp_creds_path)

if __name__ == "__main__":
    deploy_tflite_artifacts()
