import os
import json
import base64
from firebase_admin import initialize_app, storage, credentials
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TFLITE_MODEL_PATH = "face_recognition_model.tflite"
CLASS_MAPPING_PATH = "class_mapping.json"

# Same bucket your Android app is downloading from
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

def deploy_tflite_artifacts():
    """
    Uploads the TFLite model + class mapping JSON to Firebase Storage.
    This will be consumed by the Android app.
    """
    # Verify artifacts exist
    if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(CLASS_MAPPING_PATH):
        logging.error("Required artifacts missing. Did training + conversion run?")
        return

    # Load Base64 encoded creds
    service_account_base64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not service_account_base64:
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable missing.")
        return

    temp_creds_path = "firebase_creds.json"

    try:
        # ---------------------------------------------------------
        # Decode the Base64 service account JSON
        # ---------------------------------------------------------
        logging.info("Decoding Firebase service account (Base64)...")
        decoded_json = base64.b64decode(service_account_base64).decode("utf-8")
        creds_dict = json.loads(decoded_json)

        # Save decoded credentials temporarily
        with open(temp_creds_path, "w") as f:
            json.dump(creds_dict, f)

        # ---------------------------------------------------------
        # Initialize Firebase Admin SDK
        # ---------------------------------------------------------
        cred = credentials.Certificate(temp_creds_path)
        app = initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})
        bucket = storage.bucket(app=app)

        # ---------------------------------------------------------
        # Upload TFLite Model
        # ---------------------------------------------------------
        model_blob = bucket.blob(f"mobile_artifacts/{TFLITE_MODEL_PATH}")
        model_blob.upload_from_filename(TFLITE_MODEL_PATH)
        logging.info(f"✅ Uploaded: mobile_artifacts/{TFLITE_MODEL_PATH}")

        # ---------------------------------------------------------
        # Upload Class Mapping JSON
        # ---------------------------------------------------------
        mapping_blob = bucket.blob(f"mobile_artifacts/{CLASS_MAPPING_PATH}")
        mapping_blob.upload_from_filename(CLASS_MAPPING_PATH)
        logging.info(f"✅ Uploaded: mobile_artifacts/{CLASS_MAPPING_PATH}")

    except Exception as e:
        logging.error(f"Deployment failed: {e}")

    finally:
        # Cleanup temp credentials
        if os.path.exists(temp_creds_path):
            os.remove(temp_creds_path)


if __name__ == "__main__":
    deploy_tflite_artifacts()
