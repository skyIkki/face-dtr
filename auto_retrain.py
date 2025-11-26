#!/usr/bin/env python3
"""
auto_retrain.py (Hybrid A + B)

Pipeline:
1. Download employee videos from Firebase Storage
2. Extract padded face crops via MediaPipe Face Detection
3. Optional Face Mesh alignment (best-effort)
4. Blur filtering, resize, augmentation
5. Generate 128-D normalized embeddings (MobileNetV2)
6. Average embeddings per employee
7. Save artifacts under ./mobile_artifacts/
"""

# -----------------------------
# Imports
# -----------------------------
import os
import json
import base64
import logging
import shutil
import math
import numpy as np
import cv2
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, storage as fb_storage
import mediapipe as mp
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Config
# -----------------------------
FIREBASE_PREFIX = "video_training_data/"
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

# -----------------------------
# MediaPipe Models (Initialized once)
# -----------------------------
face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.45
)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# -----------------------------
# Firebase Initialization
# -----------------------------
def initialize_firebase():
    if firebase_admin._apps:
        return

    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("Missing FIREBASE_SERVICE_ACCOUNT_KEY secret.")
        raise SystemExit(1)

    svc_json = base64.b64decode(svc_b64).decode("utf-8")
    cred = credentials.Certificate(json.loads(svc_json))
    firebase_admin.initialize_app(cred)
    logging.info("🔥 Firebase initialized.")

# -----------------------------
# Download videos from Firebase
# -----------------------------
def download_videos_from_firebase():
    initialize_firebase()
    bucket = fb_storage.bucket()

    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
    os.makedirs(USER_VIDEO_DIR, exist_ok=True)

    employee_ids = set()
    saved = 0

    for blob in bucket.list_blobs(prefix=FIREBASE_PREFIX):
        if not blob.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        relative = blob.name.replace(FIREBASE_PREFIX, "", 1)
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
            saved += 1
        except Exception as e:
            logging.error(f"❌ Download failed: {e}")

    logging.info(f"✅ {saved} videos downloaded for {len(employee_ids)} employees.")
    return sorted(employee_ids)

# -----------------------------
# Image Helpers
# -----------------------------
def rotate_image(img, angle, center=None):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def compute_blur_score(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

# -----------------------------
# Extract Face Crops (Detection + Alignment + Augmentation)
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

        if not os.path.isdir(vid_dir):
            logging.warning(f"⚠ No video folder for {emp_id} → skipping.")
            continue

        logging.info(f"🧑 Processing: {emp_id}")

        for file in os.listdir(vid_dir):
            if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            cap = cv2.VideoCapture(os.path.join(vid_dir, file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            if frame_count == 0:
                logging.debug("  ⏭ Empty video → skip")
                cap.release()
                continue

            frame_idxs = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO, dtype=int)

            for i, fidx in enumerate(frame_idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection
                results = face_detector.process(rgb)
                if not results.detections:
                    continue

                bbox = results.detections[0].location_data.relative_bounding_box
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                if bw * bh < MIN_FACE_PIXEL_AREA:
                    continue

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                px = int(bw * PAD_RATIO)
                py = int(bh * PAD_RATIO)
                x1, y1 = max(0, x - px), max(0, y - py)
                x2, y2 = min(w, x + bw + px), min(h, y + bh + py)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Blur filtering
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = compute_blur_score(gray)
                if blur < BLUR_THRESHOLD:
                    continue

                # Optional alignment
                aligned = crop
                try:
                    mesh_res = face_mesh.process(rgb)
                    if mesh_res.multi_face_landmarks:
                        left = mesh_res.multi_face_landmarks[0].landmark[33]
                        right = mesh_res.multi_face_landmarks[0].landmark[263]
                        clx = int(left.x * w) - x1
                        cly = int(left.y * h) - y1
                        crx = int(right.x * w) - x1
                        cry = int(right.y * h) - y1
                        dx, dy = crx - clx, cry - cly
                        if abs(dx) > 1e-3:
                            ang = math.degrees(math.atan2(dy, dx))
                            temp = rotate_image(crop, ang)
                            gray2 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                            if compute_blur_score(gray2) >= BLUR_THRESHOLD:
                                aligned = temp
                except Exception:
                    pass  # best-effort

                # Resize
                try:
                    final_face = cv2.resize(aligned, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue

                # Save crop
                base_name = f"{os.path.splitext(file)[0]}_f{fidx}_{i+1}"
                out_path = os.path.join(out_dir, f"{base_name}.jpg")
                cv2.imwrite(out_path, final_face)
                total_saved += 1

                # Save flipped version if enabled
                if AUGMENTATIONS:
                    cv2.imwrite(os.path.join(out_dir, f"{base_name}_flip.jpg"), cv2.flip(final_face, 1))

            cap.release()

    logging.info(f"📸 {total_saved} face images saved.")

# -----------------------------
# Build MobileNetV2 Embedding Model
# -----------------------------
def build_embedding_model():
    base = MobileNetV2(input_shape=(*TARGET_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    model = models.Model(base.input, x)
    logging.info("✅ Embedding model ready.")
    return model

# -----------------------------
# Generate and save embeddings
# -----------------------------
def generate_embeddings(model):
    embeddings = {}

    for emp_id in os.listdir(USER_DATA_DIR):
        folder = os.path.join(USER_DATA_DIR, emp_id)
        if not os.path.isdir(folder):
            continue

        vectors = []
        for file in os.listdir(folder):
            if not file.lower().endswith(".jpg"):
                continue
            img = cv2.resize(cv2.imread(os.path.join(folder, file)), TARGET_SIZE)
            img = np.expand_dims(img_to_array(img) / 255.0, 0)
            vectors.append(model.predict(img, verbose=0)[0])

        if vectors:
            embeddings[emp_id] = np.mean(np.vstack(vectors), 0)

    return embeddings

def save_class_mapping(ids):
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump({str(i): e for i, e in enumerate(ids)}, f, indent=2)

def save_embeddings_json(e):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({k: v.tolist() for k, v in e.items()}, f, indent=2)

def cleanup_temp_dirs():
    for d in (USER_VIDEO_DIR, USER_DATA_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)

    emp_ids = download_videos_from_firebase()
    save_class_mapping(emp_ids)
    extract_frames_from_videos(emp_ids)

    model = build_embedding_model()
    embeddings = generate_embeddings(model)

    if embeddings:
        save_embeddings_json(embeddings)
        model.save(MODEL_SAVE_PATH)
        logging.info("🚀 Artifacts saved for mobile.")
    else:
        logging.error("❌ No embeddings generated!")

    cleanup_temp_dirs()
