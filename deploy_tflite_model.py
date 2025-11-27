#!/usr/bin/env python3
import os
import json
import base64
import logging
import shutil
from firebase_admin import initialize_app, storage, credentials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Artifact paths
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
TFLITE_MODEL_NAME = "face_embedding_model.tflite"
CLASS_MAPPING_NAME = "class_mapping.json"
EMBEDDINGS_JSON_NAME = "employee_embeddings.json"
VERSION_FILE_PATH = "model_version.txt"

TFLITE_MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, TFLITE_MODEL_NAME)
CLASS_MAPPING_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, CLASS_MAPPING_NAME)
EMBEDDINGS_JSON_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, EMBEDDINGS_JSON_NAME)

FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

def update_model_version(filepath=VERSION_FILE_PATH):
    current_version = 0
    if os.path.exists(filepath):
        with open(filepath,'r') as f:
            content = f.read().strip()
            if content:
                current_version = int(content)
    new_version = current_version + 1
    with open(filepath,'w') as f:
        f.write(str(new_version))
    logging.info(f"Model version updated to {new_version}")

def deploy_tflite_artifacts():
    required_files = [TFLITE_MODEL_PATH, CLASS_MAPPING_PATH, EMBEDDINGS_JSON_PATH]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        logging.error(f"Missing artifacts: {missing}")
        raise FileNotFoundError(f"Cannot deploy, missing {missing}")

    service_account_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not service_account_b64:
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY not set")
        raise ValueError("Missing Firebase key.")

    temp_path = "firebase_creds.json"
    try:
        decoded = base64.b64decode(service_account_b64).decode("utf-8")
        with open(temp_path, "w") as f:
            json.dump(json.loads(decoded), f)

        cred = credentials.Certificate(temp_path)
        try:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name="deploy_app")
        except ValueError:
            app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME}, name=None)

        bucket = storage.bucket(app=app)

        def upload(local_path, remote_name):
            blob = bucket.blob(f"mobile_artifacts/{remote_name}")
            blob.upload_from_filename(local_path)
            logging.info(f"✅ Uploaded {remote_name}")

        upload(TFLITE_MODEL_PATH, TFLITE_MODEL_NAME)
        upload(CLASS_MAPPING_PATH, CLASS_MAPPING_NAME)
        upload(EMBEDDINGS_JSON_PATH, EMBEDDINGS_JSON_NAME)

        # Update version and upload
        update_model_version()
        if os.path.exists(VERSION_FILE_PATH):
            upload(VERSION_FILE_PATH, os.path.basename(VERSION_FILE_PATH))

        logging.info("🎉 All artifacts deployed successfully!")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    try:
        deploy_tflite_artifacts()
    except Exception as e:
        logging.error(f"Deployment failed: {e}")
