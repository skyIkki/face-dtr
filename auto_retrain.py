#!/usr/bin/env python3
"""
auto_retrain.py (Face-Cropped Version)
- Downloads videos from Firebase.
- Extracts ONLY FACE crops using MediaPipe.
- Saves cropped faces to user_training_data/<employee_id>/.
- Builds MobileNetV2 embedding extractor.
- Generates 128D averaged embeddings per employee.
- Saves artifacts for Android: model + embeddings + class mapping.
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

# NEW — MediaPipe for face detection
import mediapipe as mp
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIREBASE_PREFIX = "video_training_data/"
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

USER_VIDEO_DIR = "user_videos_temp"
USER_DATA_DIR = "user_training_data"

MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
MODEL_SAVE_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.h5")
EMBEDDINGS_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "employee_embeddings.json")
CLASS_MAPPING_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "class_mapping.json")

TARGET_SIZE = (160, 160)
NUM_SAMPLES_PER_VIDEO = 10

# -----------------------------
# Firebase init
# -----------------------------
def initialize_firebase():
    if firebase_admin._apps:
        return
    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("Missing FIREBASE_SERVICE_ACCOUNT_KEY environment variable.")
        raise SystemExit(1)
    svc_json = base64.b64decode(svc_b64).decode("utf-8")
    cred = credentials.Certificate(json.loads(svc_json))
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})
    logging.info("Firebase initialized.")

# -----------------------------
# Download videos
# -----------------------------
def download_videos_from_firebase():
    initialize_firebase()
    bucket = fb_storage.bucket()

    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
    os.makedirs(USER_VIDEO_DIR, exist_ok=True)

    blobs = bucket.list_blobs(prefix=FIREBASE_PREFIX)
    employee_ids = set()
    total = 0

    for blob in blobs:
        if not blob.name.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        relative = blob.name[len(FIREBASE_PREFIX):]
        parts = relative.split("/")
        if len(parts) < 2:
            continue

        emp_id = parts[0]
        employee_ids.add(emp_id)

        local_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        os.makedirs(local_dir, exist_ok=True)

        local_path = os.path.join(local_dir, parts[-1])
        blob.download_to_filename(local_path)
        total += 1

    logging.info(f"Downloaded {total} videos across {len(employee_ids)} employees.")
    return sorted(employee_ids)

# -----------------------------
# FACE-ONLY Extractor
# -----------------------------
def extract_frames_from_videos(employee_ids):
    """Extracts *only the face* from frames using MediaPipe."""
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    total_faces = 0

    for emp_id in employee_ids:
        vid_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        out_dir = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(out_dir, exist_ok=True)

        for video_file in os.listdir(vid_dir):
            if not video_file.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            path = os.path.join(vid_dir, video_file)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logging.warning(f"Failed to open {path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                continue

            indices = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO + 2, dtype=int)[1:-1]

            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # --- FACE DETECTION ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detector.process(rgb)
                if not result.detections:
                    continue

                det = result.detections[0]
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)

                face_crop = frame[y:y+height, x:x+width]
                if face_crop.size == 0:
                    continue

                save_path = os.path.join(out_dir, f"{os.path.splitext(video_file)[0]}_face_{i+1}.jpg")
                cv2.imwrite(save_path, face_crop)
                total_faces += 1

            cap.release()

    logging.info(f"Extracted and saved {total_faces} FACE crops total.")

# -----------------------------
# Embedding model
# -----------------------------
def build_embedding_model():
    base = MobileNetV2(input_shape=TARGET_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    model = models.Model(inputs=base.input, outputs=x)
    return model

# -----------------------------
# Embeddings
# -----------------------------
def generate_embeddings(model):
    emp_dirs = [d for d in os.listdir(USER_DATA_DIR) if os.path.isdir(os.path.join(USER_DATA_DIR, d))]
    embeddings = {}

    for emp_id in emp_dirs:
        folder = os.path.join(USER_DATA_DIR, emp_id)
        vectors = []

        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = load_img(img_path, target_size=TARGET_SIZE)
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            emb = model.predict(arr, verbose=0)[0]
            vectors.append(emb)

        if len(vectors) > 0:
            embeddings[emp_id] = np.mean(np.vstack(vectors), axis=0)

    return embeddings

# -----------------------------
# Save artifacts
# -----------------------------
def save_class_mapping(employee_ids):
    mapping = {str(i): emp for i, emp in enumerate(employee_ids)}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)

def save_embeddings_json(emb_dict):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({k: v.tolist() for k, v in emb_dict.items()}, f, indent=4)

# -----------------------------
# Cleanup
# -----------------------------
def cleanup_temp_dirs():
    for d in (USER_VIDEO_DIR, USER_DATA_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)

    try:
        emp_ids = download_videos_from_firebase()
        if not emp_ids:
            logging.error("No employee videos found.")
            raise SystemExit(1)

        save_class_mapping(emp_ids)
        extract_frames_from_videos(emp_ids)

        model = build_embedding_model()
        embeddings = generate_embeddings(model)

        save_embeddings_json(embeddings)
        model.save(MODEL_SAVE_PATH)

        logging.info("Training complete — artifacts ready for TFLite conversion.")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

    finally:
        cleanup_temp_dirs()
