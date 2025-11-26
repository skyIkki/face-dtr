#!/usr/bin/env python3
"""
auto_retrain.py (Hybrid A+B)
- Downloads videos from Firebase.
- Extracts face crops using MediaPipe Face Detection.
- Attempts Face Mesh alignment for higher accuracy (best-effort).
- Adds padding, blur filtering, resizing, augmentation.
- Builds MobileNetV2 embedding extractor (128-D normalized).
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

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# MediaPipe (face detection + face mesh)
import mediapipe as mp
mp_face_det = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize detectors once for reuse
face_detector = mp_face_det.FaceDetection(model_selection=1, min_detection_confidence=0.45)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIREBASE_PREFIX = "video_training_data/"
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.appspot.com"  # ✅ Fixed bucket name

USER_VIDEO_DIR = "user_videos_temp"
USER_DATA_DIR = "user_training_data"

MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
MODEL_SAVE_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_embedding_model.h5")
EMBEDDINGS_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "employee_embeddings.json")
CLASS_MAPPING_FILE = os.path.join(MOBILE_ARTIFACTS_DIR, "class_mapping.json")

TARGET_SIZE = (160, 160)
NUM_SAMPLES_PER_VIDEO = 10
PAD_RATIO = 0.35
BLUR_THRESHOLD = 40.0
AUGMENTATIONS = True
MIN_FACE_PIXEL_AREA = 32 * 32

# Firebase initialization
def initialize_firebase():
    if firebase_admin._apps:
        return
    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("Missing FIREBASE_SERVICE_ACCOUNT_KEY environment variable.")
        raise SystemExit(1)

    svc_json = base64.b64decode(svc_b64).decode("utf-8")
    cred = credentials.Certificate(json.loads(svc_json))

    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})  # ✅ Using correct config
    logging.info("Firebase initialized.")

# Download employee videos
def download_videos_from_firebase():
    initialize_firebase()
    bucket = fb_storage.bucket(FIREBASE_BUCKET_NAME)  # ✅ Explicit bucket reference

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
            logging.error(f"Download failed {blob.name}: {e}")

    logging.info(f"Downloaded {total} videos for {len(employee_ids)} employees.")
    return sorted(employee_ids)

# Rotate image helper
def rotate_image(img, angle, center=None):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Blur score function
def compute_blur_score(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

# Face mesh alignment (best-effort)
def mesh_align_crop(crop_bgr, global_bbox, mesh_result):
    try:
        if mesh_result is None or not mesh_result.multi_face_landmarks:
            return crop_bgr

        lm = mesh_result.multi_face_landmarks[0]
        left = lm.landmark[33]
        right = lm.landmark[263]

        h, w = mesh_result.image.shape[:2]
        abs_left = (int(left.x * w), int(left.y * h))
        abs_right = (int(right.x * w), int(right.y * h))

        x1, y1, x2, y2 = global_bbox
        le = (abs_left[0] - x1, abs_left[1] - y1)
        re = (abs_right[0] - x1, abs_right[1] - y1)

        dx, dy = re[0] - le[0], re[1] - le[1]
        if abs(dx) < 1e-3:
            return crop_bgr

        angle = math.degrees(math.atan2(dy, dx))
        return rotate_image(crop_bgr, angle)
    except Exception:
        return crop_bgr

# Extract face crops from sampled frames
def extract_frames_from_videos(employee_ids):
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    saved = 0
    for emp_id in employee_ids:
        src = os.path.join(USER_VIDEO_DIR, emp_id)
        dst = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(dst, exist_ok=True)

        for video_file in sorted(os.listdir(src)):
            path = os.path.join(src, video_file)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue

            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if count == 0:
                cap.release()
                continue

            idxs = np.linspace(0, count - 1, NUM_SAMPLES_PER_VIDEO + 2, dtype=int)[1:-1]
            for i, frame_i in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det = face_detector.process(rgb)

                if not det.detections:
                    continue

                box = det.detections[0].location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)

                if bw * bh < MIN_FACE_PIXEL_AREA:
                    continue

                pad_x, pad_y = int(bw * PAD_RATIO), int(bh * PAD_RATIO)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if compute_blur_score(gray) < BLUR_THRESHOLD:
                    continue

                mesh = face_mesh.process(rgb)
                crop = mesh_align_crop(crop, (x1, y1, x2, y2), mesh)

                face = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                name = f"{os.path.splitext(video_file)[0]}_f{frame_i}_face_{i+1}"
                cv2.imwrite(os.path.join(dst, f"{name}.jpg"), face)
                saved += 1

                if AUGMENTATIONS:
                    cv2.imwrite(os.path.join(dst, f"{name}_flip.jpg"), cv2.flip(face, 1))

            cap.release()

    logging.info(f"Saved {saved} face crops.")

# Build embedding extractor model
def build_embedding_model():
    base = MobileNetV2(input_shape=TARGET_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    out = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(128)(out)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(out)
    model = models.Model(base.input, out)
    return model

# Generate employee embeddings by averaging face vectors
def generate_embeddings(model):
    embeddings = {}
    total = 0

    for emp_id in os.listdir(USER_DATA_DIR):
        folder = os.path.join(USER_DATA_DIR, emp_id)
        if not os.path.isdir(folder):
            continue

        vecs = []
        for img in os.listdir(folder):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, img)
            try:
                arr = img_to_array(load_img(path, target_size=TARGET_SIZE)) / 255.0
                vecs.append(model.predict(np.expand_dims(arr, 0), verbose=0)[0])
                total += 1
            except Exception:
                pass

        if vecs:
            embeddings[emp_id] = np.mean(np.vstack(vecs), 0)

    logging.info(f"Embedding generated from {total} images.")
    return embeddings

# Save employee class mapping
def save_class_mapping(employee_ids):
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump({str(i): e for i, e in enumerate(employee_ids)}, f, indent=2)

# Save embeddings JSON
def save_embeddings_json(emb_dict):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({k: v.tolist() for k, v in emb_dict.items()}, f, indent=2)

# Cleanup temporary directories
def cleanup_temp_dirs():
    for d in (USER_VIDEO_DIR, USER_DATA_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)

# Run pipeline
if __name__ == "__main__":
    os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)

    emp_ids = download_videos_from_firebase()
    save_class_mapping(emp_ids)
    extract_frames_from_videos(emp_ids)

    model = build_embedding_model()
    embeddings = generate_embeddings(model)

    save_embeddings_json(embeddings)
    model.save(MODEL_SAVE_PATH)

    cleanup_temp_dirs()
