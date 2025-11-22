#!/usr/bin/env python3
"""
auto_retrain.py
- Downloads videos from Firebase under prefix "video_training_data/"
  expecting structure: video_training_data/<employee_id>/<videofile>.mp4
- Extracts frames (uniform samples) to user_training_data/<employee_id>/
- Trains MobileNetV2-based softmax classifier
- Saves face_recognition_model.h5 and class_mapping.json (string keys)
- Leaves artifacts for conversion/deployment; only removes temp dirs
"""

import os
import json
import base64
import logging
import shutil
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIREBASE_PREFIX = "video_training_data/"           # where videos are in the bucket
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"  # change if needed

USER_VIDEO_DIR = "user_videos_temp"               # temp downloaded raw videos
USER_DATA_DIR = "user_training_data"              # extracted frames for training (folders per employee)
MODEL_SAVE_PATH = "face_recognition_model.h5"
CLASS_MAPPING_FILE = "class_mapping.json"

TARGET_SIZE = (160, 160)
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_SAMPLES_PER_VIDEO = 10   # sample uniformly from the video (20s -> pick 10 frames)
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

# -----------------------------
# Firebase init (expects base64-encoded JSON in env var)
# -----------------------------
def initialize_firebase():
    if firebase_admin._apps:
        return
    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable missing (base64 JSON expected).")
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
        # skip directory placeholders
        if blob.name.endswith('/') or not blob.name.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        # path like video_training_data/<employee_id>/<file>
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
            logging.debug(f"Downloaded {blob.name} -> {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")
    logging.info(f"Downloaded {downloaded} videos across {len(employee_ids)} employees.")
    return sorted(employee_ids)

# -----------------------------
# Extract frames uniformly per video into USER_DATA_DIR/<emp_id>/
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
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if frame_count == 0:
                logging.warning(f"No frames in {path}")
                cap.release()
                continue

            # sample NUM_SAMPLES_PER_VIDEO indices uniformly (avoid first/last frames if noisy)
            indices = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO + 2, dtype=int)[1:-1]
            saved = 0
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                # optional: convert BGR->RGB
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_f{i+1}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
            cap.release()
            total_frames += saved
            logging.info(f"Extracted {saved} frames from {path}")

    logging.info(f"Finished extracting frames: {total_frames} frames from {total_videos} videos.")

# -----------------------------
# Build the classifier model
# -----------------------------
def build_classifier(num_classes):
    base = MobileNetV2(input_shape=TARGET_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    logging.info("Built MobileNetV2 classifier head (softmax).")
    return model

# -----------------------------
# Train from USER_DATA_DIR folder structure
# -----------------------------
def train_from_images():
    # Check data exists
    if not os.path.exists(USER_DATA_DIR) or not any(os.scandir(USER_DATA_DIR)):
        logging.critical(f"No training images found in {USER_DATA_DIR}")
        raise SystemExit(1)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    train_gen = datagen.flow_from_directory(
        USER_DATA_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_gen = datagen.flow_from_directory(
        USER_DATA_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=RANDOM_SEED
    )

    num_classes = len(train_gen.class_indices)
    logging.info(f"Training images: {train_gen.samples}, validation images: {val_gen.samples}, classes: {num_classes}")
    if num_classes < 1:
        logging.critical("Not enough classes to train.")
        raise SystemExit(1)

    model = build_classifier(num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS, verbose=2)
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Saved trained model to {MODEL_SAVE_PATH}")

    # Save mapping: map index -> employee_id but keys as strings
    # train_gen.class_indices is {label_string: index}
    class_indices = train_gen.class_indices  # label -> index (int)
    inverse_map = {str(index): label for label, index in class_indices.items()}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(inverse_map, f, indent=4)
    logging.info(f"Saved class mapping to {CLASS_MAPPING_FILE} (string keys).")
    return model

# -----------------------------
# Minimal cleanup (do NOT delete .h5/.json)
# -----------------------------
def cleanup_temp_dirs():
    for p in (USER_VIDEO_DIR, ):
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
    try:
        emp_ids = download_videos_from_firebase()
        if len(emp_ids) == 0:
            logging.critical("No employee videos found. Exiting.")
            raise SystemExit(1)

        extract_frames_from_videos(emp_ids)
        train_from_images()
        logging.info("Training pipeline finished. Model and mapping are preserved for conversion/deployment.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
    finally:
        # cleanup only temporary raw videos
        cleanup_temp_dirs()
