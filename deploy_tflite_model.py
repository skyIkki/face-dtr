import os
import json
import base64
from firebase_admin import initialize_app, storage, credentials
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CORRECTED PATHS ---
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"

# Note: Correcting the TFLite model name to match the output of convert_to_tflite.py
TFLITE_MODEL_NAME = "face_embedding_model.tflite"
CLASS_MAPPING_NAME = "class_mapping.json"
EMBEDDINGS_JSON_NAME = "employee_embeddings.json" # New artifact to deploy

# Full local paths
TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, TFLITE_MODEL_NAME)
CLASS_MAPPING_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, CLASS_MAPPING_NAME)
EMBEDDINGS_JSON_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, EMBEDDINGS_JSON_NAME)

FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

def deploy_tflite_artifacts():
    """
    Uploads the TFLite model, class mapping JSON, and employee embeddings to Firebase Storage.
    This is consumed by the Android app.
    """
    required_files = {
        TFLITE_MODEL_PATH, 
        CLASS_MAPPING_PATH, 
        EMBEDDINGS_JSON_PATH
    }
    
    # Verify all artifacts exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logging.error(f"Required artifacts missing: {', '.join(missing_files)}. Did all preceding steps run successfully?")
        return

    # Load Base64 encoded creds
    service_account_base64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not service_account_base64:
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable missing.")
        return

    temp_creds_path = "firebase_creds.json"

    try:
        # ---------------------------------------------------------
        # Decode the Base64 service account JSON and save temporarily
        # ---------------------------------------------------------
        logging.info("Decoding Firebase service account (Base64)...")
        decoded_json = base64.b64decode(service_account_base64).decode("utf-8")
        creds_dict = json.loads(decoded_json)

        with open(temp_creds_path, "w") as f:
            json.dump(creds_dict, f)

        # ---------------------------------------------------------
        # Initialize Firebase Admin SDK
        # ---------------------------------------------------------
        cred = credentials.Certificate(temp_creds_path)
        # Use try/except to handle case where app might already be initialized (though less common in scripts)
        try:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})
        except ValueError:
            # If the app is already initialized, just get the default one
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name="deploy_app")
        
        bucket = storage.bucket(app=app)

        # ---------------------------------------------------------
        # Deployment Helper
        # ---------------------------------------------------------
        def upload_artifact(local_path, destination_name):
            blob_path = f"mobile_artifacts/{destination_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logging.info(f"✅ Uploaded: {blob_path}")

        # ---------------------------------------------------------
        # Upload Artifacts
        # ---------------------------------------------------------
        upload_artifact(TFLITE_MODEL_PATH, TFLITE_MODEL_NAME)
        upload_artifact(CLASS_MAPPING_PATH, CLASS_MAPPING_NAME)
        upload_artifact(EMBEDDINGS_JSON_PATH, EMBEDDINGS_JSON_NAME)
        
        logging.info("🎉 All mobile artifacts deployed successfully!")


    except Exception as e:
        logging.error(f"Deployment failed: {e}")

    finally:
        # Cleanup temp credentials
        if os.path.exists(temp_creds_path):
            os.remove(temp_creds_path)
            
# -----------------------------
# Versioning Function
# -----------------------------
VERSION_FILE_PATH = "model_version.txt" # Defined in Config section

def update_model_version(filepath=VERSION_FILE_PATH):
    """Reads, increments, and writes the new model version."""
    current_version = 0
    
    # 1. Read Current Version
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    current_version = int(content)
        else:
            logging.warning(f"{filepath} not found. Starting version from 0.")
    except Exception as e:
        logging.error(f"Error reading version file, cannot update: {e}")
        return

    # 2. Increment and Write New Version
    new_version = current_version + 1
    
    try:
        with open(filepath, 'w') as f:
            f.write(str(new_version))
        logging.info(f"Successfully updated model_version.txt to {new_version}.")
    except Exception as e:
        logging.critical(f"FATAL: Failed to write new version to {filepath}: {e}")
        raise # Fail the pipeline if the version can't be saved

if __name__ == "__main__":
    deploy_tflite_artifacts()
