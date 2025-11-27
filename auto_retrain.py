#!/usr/bin/env python3
"""
auto_retrain.py (Enhanced Hybrid A+B)
- Downloads videos from Firebase.
- Extracts face crops using MediaPipe Face Detection.
- Attempts Face Mesh alignment for higher accuracy (best-effort).
- Adds padding, blur filtering, resizing, and rich augmentations.
- Builds MobileNetV2 embedding extractor (128-D normalized, optional fine-tuning).
- Averages embeddings per employee and saves artifacts under mobile_artifacts/.
"""

import os
import json
import base64
import logging
import shutil
import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# MediaPipe (face detection + face mesh)
import mediapipe as mp
mp_face_det = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe once
face_detector = mp_face_det.FaceDetection(model_selection=1, min_detection_confidence=0.45)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.3,
                                  min_tracking_confidence=0.3)

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

TARGET_SIZE = (160, 160)      # width, height used for model input
NUM_SAMPLES_PER_VIDEO = 10    # frames sampled per video
PAD_RATIO = 0.35              # padding around detected bbox
BLUR_THRESHOLD = 40.0         # Laplacian variance threshold
AUGMENTATIONS = True
MIN_FACE_PIXEL_AREA = 32*32

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
        if not blob.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
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
        try:
            blob.download_to_filename(local_path)
            total += 1
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")

    logging.info(f"Downloaded {total} videos across {len(employee_ids)} employees.")
    return sorted(employee_ids)

# -----------------------------
# Utility helpers
# -----------------------------
def rotate_image(img, angle, center=None):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def compute_blur_score(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def mesh_align_crop(crop_bgr, global_bbox, mesh_result):
    try:
        if mesh_result is None or not mesh_result.multi_face_landmarks:
            return crop_bgr
        lm = mesh_result.multi_face_landmarks[0]
        left_idx, right_idx = 33, 263
        ih, iw = mesh_result.image.shape[:2]
        left = lm.landmark[left_idx]
        right = lm.landmark[right_idx]
        abs_left_x = int(left.x * iw); abs_left_y = int(left.y * ih)
        abs_right_x = int(right.x * iw); abs_right_y = int(right.y * ih)
        x1, y1, x2, y2 = global_bbox
        le_x = abs_left_x - x1; le_y = abs_left_y - y1
        re_x = abs_right_x - x1; re_y = abs_right_y - y1
        dx = re_x - le_x; dy = re_y - le_y
        if abs(dx) < 1e-3:
            return crop_bgr
        angle = math.degrees(math.atan2(dy, dx))
        aligned = rotate_image(crop_bgr, angle, center=(crop_bgr.shape[1]//2, crop_bgr.shape[0]//2))
        return aligned
    except Exception:
        return crop_bgr

# -----------------------------
# Augmentation functions
# -----------------------------
def augment_image(img):
    augmented = [img]
    h, w = img.shape[:2]
    # horizontal flip
    augmented.append(cv2.flip(img, 1))
    # small rotation ±15°
    angle = np.random.uniform(-15, 15)
    augmented.append(rotate_image(img, angle))
    # brightness/contrast
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.randint(-15, 15)
    bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    augmented.append(bright)
    # random zoom ±10%
    zx, zy = np.random.uniform(0.9, 1.1, 2)
    zx = int(zx*w); zy=int(zy*h)
    zoom_img = cv2.resize(img, (zx, zy))
    zoom_img = cv2.resize(zoom_img, (w, h))
    augmented.append(zoom_img)
    return augmented

# -----------------------------
# FACE-ONLY Extractor
# -----------------------------
def extract_frames_from_videos(employee_ids):
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    total_saved = 0
    for emp_id in employee_ids:
        vid_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        out_dir = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"Processing employee: {emp_id}")
        if not os.path.exists(vid_dir):
            logging.warning(f"No video directory found for {emp_id}, skipping.")
            continue
        for video_file in sorted(os.listdir(vid_dir)):
            if not video_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            video_path = os.path.join(vid_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            indices = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO+2, dtype=int)[1:-1]
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_result = face_detector.process(rgb)
                if not det_result.detections:
                    continue
                det = det_result.detections[0]
                rel_bbox = det.location_data.relative_bounding_box
                x, y = int(rel_bbox.xmin * w), int(rel_bbox.ymin * h)
                bw, bh = int(rel_bbox.width * w), int(rel_bbox.height * h)
                if bw*bh < MIN_FACE_PIXEL_AREA:
                    continue
                pad_x, pad_y = int(bw*PAD_RATIO), int(bh*PAD_RATIO)
                x1, y1, x2, y2 = max(0, x-pad_x), max(0, y-pad_y), min(w, x+bw+pad_x), min(h, y+bh+pad_y)
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr is None or crop_bgr.size==0:
                    continue
                gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                if compute_blur_score(gray_crop) < BLUR_THRESHOLD:
                    continue
                try:
                    mesh_result = face_mesh.process(rgb)
                    aligned_crop = mesh_align_crop(crop_bgr, (x1, y1, x2, y2), mesh_result)
                except Exception:
                    aligned_crop = crop_bgr
                try:
                    resized = cv2.resize(aligned_crop, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue
                # Save augmented crops
                crops_to_save = augment_image(resized) if AUGMENTATIONS else [resized]
                for j, img_aug in enumerate(crops_to_save):
                    out_path = os.path.join(out_dir, f"{os.path.splitext(video_file)[0]}_f{idx}_{i+1}_{j}.jpg")
                    try:
                        cv2.imwrite(out_path, img_aug)
                        total_saved += 1
                    except Exception:
                        continue
            cap.release()
    logging.info(f"Extracted and saved {total_saved} face crops total under '{USER_DATA_DIR}'.")

# -----------------------------
# Embedding model
# -----------------------------
def build_embedding_model(fine_tune=False):
    base = tf.keras.applications.MobileNetV2(input_shape=TARGET_SIZE+(3,), include_top=False, weights='imagenet')
    if not fine_tune:
        base.trainable = False
    else:
        # Fine-tune last 10 layers
        for layer in base.layers[:-10]:
            layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    model = models.Model(inputs=base.input, outputs=x)
    logging.info("Built MobileNetV2 embedding model.")
    return model

# -----------------------------
# Generate embeddings
# -----------------------------
def generate_embeddings(model):
    emp_dirs = [d for d in os.listdir(USER_DATA_DIR) if os.path.isdir(os.path.join(USER_DATA_DIR, d))]
    embeddings = {}
    total_images = 0
    for emp_id in emp_dirs:
        folder = os.path.join(USER_DATA_DIR, emp_id)
        vectors = []
        for img_file in os.listdir(folder):
            if not img_file.lower().endswith((".jpg",".jpeg",".png")):
                continue
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=TARGET_SIZE)
                arr = img_to_array(img)/127.5 - 1.0
                arr = np.expand_dims(arr, axis=0)
                emb = model.predict(arr, verbose=0)[0].astype(np.float32)
                vectors.append(emb)
                total_images += 1
            except Exception:
                continue
        if len(vectors) > 0:
            embeddings[emp_id] = np.mean(np.vstack(vectors), axis=0)
    logging.info(f"Generated embeddings for {len(embeddings)} employees from {total_images} images.")
    return embeddings

# -----------------------------
# Save artifacts
# -----------------------------
def save_class_mapping(employee_ids):
    mapping = {str(i): emp for i, emp in enumerate(employee_ids)}
    with open(CLASS_MAPPING_FILE,"w") as f:
        json.dump(mapping, f, indent=4)
    logging.info(f"Saved class mapping to {CLASS_MAPPING_FILE}")

def save_embeddings_json(emb_dict):
    with open(EMBEDDINGS_FILE,"w") as f:
        json.dump({k:v.tolist() for k,v in emb_dict.items()}, f, indent=4)
    logging.info(f"Saved embeddings JSON to {EMBEDDINGS_FILE}")

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
            logging.error("No employee videos found; exiting.")
            raise SystemExit(1)
        save_class_mapping(emp_ids)
        extract_frames_from_videos(emp_ids)
        model = build_embedding_model(fine_tune=False)
        embeddings = generate_embeddings(model)
        if not embeddings:
            logging.critical("No embeddings generated; aborting save.")
            raise SystemExit(1)
        save_embeddings_json(embeddings)
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Saved embedding model to {MODEL_SAVE_PATH}. Ready for TFLite conversion.")
    finally:
        cleanup_temp_dirs()
