#!/usr/bin/env python3
"""
auto_retrain.py (Embeddings version)
- Downloads videos from Firebase.
- Extracts frames to user_training_data/<employee_id>/.
- Builds MobileNetV2 embedding extractor.
- Generates 128D embeddings per employee.
- Saves ALL necessary artifacts (Model, Embeddings, Mapping) to mobile_artifacts/.
- Temporary directories are cleaned up.
"""

import os
import json
import base64
import logging
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIREBASE_PREFIX = "video_training_data/"
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

USER_VIDEO_DIR = "user_videos_temp"
USER_DATA_DIR = "user_training_data"

# --- ARTIFACTS CONFIG (CRITICAL CHANGES) ---
MOBILE_ARTIFACTS_DIR = "mobile_artifacts" # New folder for final outputs
MODEL_SAVE_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.h5")
EMBEDDINGS_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "employee_embeddings.json") # Renamed for clarity in pipeline
CLASS_MAPPING_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "class_mapping.json") # NEW ARTIFACT FILE

TARGET_SIZE = (160, 160)
NUM_SAMPLES_PER_VIDEO = 10  # frames per video

# -----------------------------
# Firebase init
# -----------------------------
def initialize_firebase():
    if firebase_admin._apps:
        return
    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable missing.")
        raise SystemExit(1)
    try:
        svc_json = base64.b64decode(svc_b64).decode("utf-8")
        cred = credentials.Certificate(json.loads(svc_json))
        firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})
        logging.info("Firebase Admin initialized.")
    except Exception as e:
        logging.critical(f"Failed to initialize Firebase Admin: {e}")
        raise

# -----------------------------
# Download videos preserving class structure
# -----------------------------
def download_videos_from_firebase():
    initialize_firebase()
    bucket = fb_storage.bucket()
    blobs = bucket.list_blobs(prefix=FIREBASE_PREFIX)
    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
    os.makedirs(USER_VIDEO_DIR, exist_ok=True)

    employee_ids = set()
    downloaded = 0
    for blob in blobs:
        if blob.name.endswith('/') or not blob.name.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        relative = blob.name[len(FIREBASE_PREFIX):]
        parts = relative.split('/')
        if len(parts) < 2:
            logging.warning(f"Skipping blob with unexpected structure: {blob.name}")
            continue
        emp_id = parts[0]
        employee_ids.add(emp_id)
        local_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, parts[-1])
        try:
            blob.download_to_filename(local_path)
            downloaded += 1
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")
    logging.info(f"Downloaded {downloaded} videos across {len(employee_ids)} employees.")
    return sorted(list(employee_ids)) # Return list of unique employee IDs

# -----------------------------
# Extract frames uniformly per video
# -----------------------------
def extract_frames_from_videos(employee_ids):
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    total_frames = 0
    total_videos = 0

    for emp_id in employee_ids:
        vid_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        out_dir = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(vid_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            total_videos += 1
            path = os.path.join(vid_dir, fname)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logging.warning(f"Could not open {path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if frame_count == 0:
                logging.warning(f"No frames in {path}")
                cap.release()
                continue

            indices = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO + 2, dtype=int)[1:-1]
            saved = 0
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_f{i+1}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
            cap.release()
            total_frames += saved
            logging.info(f"Extracted {saved} frames from {path}")

    logging.info(f"Finished extracting frames: {total_frames} frames from {total_videos} videos.")

# -----------------------------
# Build embedding model
# -----------------------------
def build_embedding_model():
    base = MobileNetV2(input_shape=TARGET_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    model = models.Model(inputs=base.input, outputs=x)
    logging.info("Built MobileNetV2 embedding model.")
    return model

# -----------------------------
# Generate embeddings per employee
# -----------------------------
def generate_embeddings(model):
    employee_dirs = [d for d in os.listdir(USER_DATA_DIR) if os.path.isdir(os.path.join(USER_DATA_DIR, d))]
    embeddings = {}

    for emp_id in employee_dirs:
        emp_path = os.path.join(USER_DATA_DIR, emp_id)
        embeddings[emp_id] = []
        for img_file in os.listdir(emp_path):
            img_path = os.path.join(emp_path, img_file)
            img = load_img(img_path, target_size=TARGET_SIZE)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            emb = model.predict(x, verbose=0)
            embeddings[emp_id].append(emb[0])

    # Average embeddings per employee
    avg_embeddings = {emp_id: np.mean(np.vstack(embs), axis=0) for emp_id, embs in embeddings.items()}
    return avg_embeddings

# -----------------------------
# Save Artifacts
# -----------------------------
def save_class_mapping(employee_ids, filepath=CLASS_MAPPING_FILE):
    """Saves the mapping of numerical index to Employee ID, needed for the client."""
    # Mapping format: {"0": "2019-0001", "1": "2019-0002", ...}
    class_mapping = {str(i): emp_id for i, emp_id in enumerate(employee_ids)}
    with open(filepath, "w") as f:
        json.dump(class_mapping, f, indent=4)
    logging.info(f"Saved class mapping to {filepath}")

def save_embeddings_json(embeddings, filepath=EMBEDDINGS_FILE):
    """Saves the averaged embedding vectors per employee."""
    emb_dict = {emp_id: emb.tolist() for emp_id, emb in embeddings.items()}
    with open(filepath, "w") as f:
        json.dump(emb_dict, f, indent=4)
    logging.info(f"Saved embeddings to {filepath}")


# -----------------------------
# Cleanup temp dirs
# -----------------------------
def cleanup_temp_dirs():
    for p in (USER_VIDEO_DIR, USER_DATA_DIR):
        if os.path.exists(p):
            try:
                shutil.rmtree(p)
                logging.info(f"Removed temp dir: {p}")
            except Exception as e:
                logging.warning(f"Could not remove {p}: {e}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Ensure the artifact directory exists
    os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)
    
    try:
        emp_ids = download_videos_from_firebase()
        if len(emp_ids) == 0:
            logging.critical("No employee videos found. Exiting.")
            raise SystemExit(1)

        # 1. Save the class mapping before feature extraction
        save_class_mapping(emp_ids)

        # 2. Extract frames and build model
        extract_frames_from_videos(emp_ids)
        model = build_embedding_model()
        
        # 3. Generate embeddings
        embeddings = generate_embeddings(model)
        
        # 4. Save final artifacts
        save_embeddings_json(embeddings)
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Saved embedding model to {MODEL_SAVE_PATH}. Ready for TFLite conversion.")
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
    finally:
        cleanup_temp_dirs()
