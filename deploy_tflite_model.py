import os
import json
import base64
from firebase_admin import initialize_app, storage, credentials
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
TFLITE_MODEL_NAME = "face_embedding_model.tflite"
CLASS_MAPPING_NAME = "class_mapping.json"
EMBEDDINGS_JSON_NAME = "employee_embeddings.json" 
VERSION_FILE_PATH = "model_version.txt" # Defined outside mobile_artifacts for root access

# Full local paths
TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, TFLITE_MODEL_NAME)
CLASS_MAPPING_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, CLASS_MAPPING_NAME)
EMBEDDINGS_JSON_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, EMBEDDINGS_JSON_NAME)

FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

# -----------------------------
# Versioning Function (NEW)
# -----------------------------
def update_model_version(filepath=VERSION_FILE_PATH):
    """
    Reads the current version, increments it, and writes the new version back.
    This ensures the version is only incremented after a successful deployment.
    """
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
        # Re-raise the exception to fail the pipeline if the version file cannot be updated
        raise

# -----------------------------
# Deployment Function
# -----------------------------
def deploy_tflite_artifacts():
    """
    Uploads the TFLite model, class mapping JSON, and employee embeddings to Firebase Storage.
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
        # Raise an error to ensure the pipeline fails if files are missing
        raise FileNotFoundError(f"Deployment failed due to missing artifacts: {', '.join(missing_files)}")

    # Load Base64 encoded creds
    service_account_base64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not service_account_base64:
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable missing.")
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY not set.")

    temp_creds_path = "firebase_creds.json"

    try:
        # Decode the Base64 service account JSON and save temporarily
        logging.info("Decoding Firebase service account (Base64)...")
        decoded_json = base64.b64decode(service_account_base64).decode("utf-8")
        creds_dict = json.loads(decoded_json)

        with open(temp_creds_path, "w") as f:
            json.dump(creds_dict, f)

        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(temp_creds_path)
        try:
            # Use 'deploy_app' name to avoid conflicts if default app is already initialized elsewhere
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name="deploy_app")
        except ValueError:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name=None) # Use default app if available
        
        bucket = storage.bucket(app=app)

        # Deployment Helper
        def upload_artifact(local_path, destination_name):
            blob_path = f"mobile_artifacts/{destination_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logging.info(f"✅ Uploaded: {blob_path}")

        # Upload Artifacts
        upload_artifact(TFLITE_MODEL_PATH, TFLITE_MODEL_NAME)
        upload_artifact(CLASS_MAPPING_PATH, CLASS_MAPPING_NAME)
        upload_artifact(EMBEDDINGS_JSON_PATH, EMBEDDINGS_JSON_NAME)
        
        logging.info("🎉 All mobile artifacts deployed successfully!")

        # ---------------------------------------------------------
        # <<< CRITICAL STEP: INCREMENT VERSION AFTER SUCCESSFUL DEPLOYMENT >>>
        # ---------------------------------------------------------
        update_model_version()


    except Exception as e:
        logging.error(f"Deployment failed: {e}")
        # Re-raise the exception to fail the CI/CD job
        raise

    finally:
        # Cleanup temp credentials
        if os.path.exists(temp_creds_path):
            os.remove(temp_creds_path)
            
# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    try:
        deploy_tflite_artifacts()
    except Exception:
        # Catch exception from deployment and let the CI/CD job fail naturally
        pass
