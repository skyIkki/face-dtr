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
VERSION_FILE_PATH = "model_version.txt"  # Defined outside mobile_artifacts for root access

# Full local paths
TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, TFLITE_MODEL_NAME)
CLASS_MAPPING_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, CLASS_MAPPING_NAME)
EMBEDDINGS_JSON_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, EMBEDDINGS_JSON_NAME)

FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

# -----------------------------
# Versioning Function
# -----------------------------
def update_model_version(filepath=VERSION_FILE_PATH):
    """Reads the current version, increments it, writes back."""
    current_version = 0

    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    current_version = int(content)
        else:
            logging.warning(f"{filepath} not found. Starting version from 0.")
    except Exception as e:
        logging.error(f"Error reading version file: {e}")
        return

    new_version = current_version + 1

    try:
        with open(filepath, 'w') as f:
            f.write(str(new_version))
        logging.info(f"Successfully updated model_version.txt to {new_version}.")
    except Exception as e:
        logging.critical(f"FATAL: Failed to write new version: {e}")
        raise

# -----------------------------
# Deployment Function
# -----------------------------
def deploy_tflite_artifacts():
    required_files = {TFLITE_MODEL_PATH, CLASS_MAPPING_PATH, EMBEDDINGS_JSON_PATH}

    # Verify artifacts exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logging.error(f"Missing artifacts: {', '.join(missing_files)}")
        raise FileNotFoundError(f"Deployment failed: missing {', '.join(missing_files)}")

    # Decode Firebase credentials
    service_account_base64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not service_account_base64:
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY missing.")
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY not set.")

    temp_creds_path = "firebase_creds.json"

    try:
        logging.info("Decoding Firebase service account (Base64)...")
        decoded_json = base64.b64decode(service_account_base64).decode("utf-8")
        creds_dict = json.loads(decoded_json)
        with open(temp_creds_path, "w") as f:
            json.dump(creds_dict, f)

        # Initialize Firebase
        cred = credentials.Certificate(temp_creds_path)
        try:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name="deploy_app")
        except ValueError:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name=None)

        bucket = storage.bucket(app=app)

        # Helper to upload
        def upload_artifact(local_path, destination_name):
            blob_path = f"mobile_artifacts/{destination_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logging.info(f"✅ Uploaded: {blob_path}")

        # Upload main artifacts
        upload_artifact(TFLITE_MODEL_PATH, TFLITE_MODEL_NAME)
        upload_artifact(CLASS_MAPPING_PATH, CLASS_MAPPING_NAME)
        upload_artifact(EMBEDDINGS_JSON_PATH, EMBEDDINGS_JSON_NAME)
        logging.info("🎉 All mobile artifacts deployed successfully!")

        # Update version
        update_model_version()

        # Upload model_version.txt to Firebase
        if os.path.exists(VERSION_FILE_PATH):
            upload_artifact(VERSION_FILE_PATH, os.path.basename(VERSION_FILE_PATH))
        else:
            logging.warning(f"{VERSION_FILE_PATH} not found, skipping upload.")

    except Exception as e:
        logging.error(f"Deployment failed: {e}")
        raise

    finally:
        if os.path.exists(temp_creds_path):
            os.remove(temp_creds_path)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    try:
        deploy_tflite_artifacts()
    except Exception:
        pass
