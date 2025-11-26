#!/usr/bin/env python3
"""
auto_retrain.py (Hybrid A+B)

✔ Downloads employee videos from Firebase
✔ Extracts face crops using MediaPipe + optional Face Mesh alignment
✔ Applies padding, blur filtering, resizing, and augmentation
✔ Builds MobileNetV2 128-D normalized embeddings
✔ Averages embeddings per employee and saves artifacts
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
from tensorflow.keras.preprocessing.image import img_to_array, load_img, load_img
from tensorflow.keras.preprocessing.image import img_to_array
from moviepy.editor import VideoFileClip  # ✅ Works once installed correctly

# MediaPipe setup (face detection + face mesh alignment)
import mediapipe as mp
mp_face_det = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_det.FaceDetection(model_selection=1, min_detection_confidence=0.45)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ───────────────────────────
# CONFIGURATION
# ───────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s ─ %(levelname)s ─ %(message)s")

FIREBASE_PREFIX = "video_training_data/"
FIREBASE_BUCKET = "face-dtr-6efa3.firebasestorage.app"

TEMP_VIDEO_DIR = "user_videos_temp"
TEMP_IMAGE_DIR = "user_training_data"

ARTIFACTS_DIR = "mobile_artifacts"
MODEL_SAVE = os.path.join(ARTIFACTS_DIR, "face_embedding_model.h5")
EMBED_JSON = os.path.join(ARTIFACTS_DIR, "employee_embeddings.json")
CLASS_JSON = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

FACE_SIZE = (160, 160)
FRAMES_PER_VIDEO = 10
PAD_RATIO = 0.35
BLUR_LIMIT = 40.0
MIN_FACE_AREA = 32 * 32
APPLY_AUGMENT = True

# ───────────────────────────
# FIREBASE INITIALIZATION
# ───────────────────────────
def initialize_firebase():
    if firebase_admin._apps:
        return
    svc_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not svc_b64:
        logging.critical("🚨 FIREBASE_SERVICE_ACCOUNT_KEY missing!")
        raise SystemExit(1)

    svc_json = base64.b64decode(svc_b64).decode("utf-8")
    cred = credentials.Certificate(json.loads(svc_json))
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_PREFIX})
    logging.info("🔥 Firebase initialized successfully.")

# ───────────────────────────
# DOWNLOAD EMPLOYEE VIDEOS
# ───────────────────────────
def download_videos_from_firebase():
    initialize_firebase()
    bucket = storage.bucket()

    if os.path.exists(TEMP_VIDEO_DIR):
        shutil.rmtree(TEMP_VIDEO_DIR)
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

    blobs = bucket.list_blobs(prefix=FIREBASE_BUCKET)
    employees = set()
    count = 0

    for blob in blobs:
        if not blob.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        path = blob.name[len(FIREBASE_PREFIX):].split("/")
        if len(path) < 2:
            continue

        emp_id = path[0]
        employees.add(emp_id)

        emp_folder = os.path.join(TEMP_VIDEO_DIR, emp_id)
        os.makedirs(emp_folder, exist_ok=True)

        local_path = os.path.join(emp_folder, path[-1])
        try:
            blob.download_to_filename(local_path)
            count += 1
        except Exception as e:
            logging.error(f"❌ Download failed: {blob.name} → {e}")

    logging.info(f"✅ {count} videos downloaded for {len(employees)} employees.")
    return sorted(employees)

# ───────────────────────────
# IMAGE HELPERS
# ───────────────────────────
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# Optional best-effort Face Mesh alignment
def mesh_align_crop(crop_bgr, full_size, bbox):
    try:
        result = face_mesh.process(full_size)
        if not result or not result.multi_face_landmarks:
            return crop_bgr

        lm = result.multi_face_landmarks[0]
        left, right = lm.landmark[33], lm.landmark[263]

        h, w = full_size.shape[:2]
        lx, ly = int(left.x * w), int(left.y * h)
        rx, ry = int(right.x * w), int(right.y * h)

        x1, y1, x2, y2 = bbox
        le_x, le_y = lx - x1, ly - y1
        re_x, re_y = rx - x1, ry - y1

        dx, dy = re_x - le_x, re_y - le_y
        if abs(dx) < 0.001:
            return crop_bgr

        angle = math.degrees(math.atan2(dy, dx))
        return rotate_image(crop_bgr, angle)

    except Exception:
        return crop_bgr  # alignment is optional → fail safe

# ───────────────────────────
# EXTRACT & SAVE FACE CROPS
# ───────────────────────────
def extract_frames_from_videos(employee_ids):
    if os.path.exists(TEMP_IMAGE_DIR):
        shutil.rmtree(TEMP_IMAGE_DIR)
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

    saved = 0

    for emp_id in employee_ids:
        video_path = os.path.join(TEMP_VIDEO_DIR, emp_id)
        emp_folder = os.path.join(TEMP_IMAGE_DIR, emp_id)
        os.makedirs(emp_folder, exist_ok=True)

        if not os.path.exists(video_path):
            continue

        for video in os.listdir(video_path):
            if not video.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            cap = cv2.VideoCapture(os.path.join(video_path, video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            if total == 0:
                cap.release()
                continue

            frames = np.linspace(0, total - 1, FRAMES_PER_VIDEO + 2, dtype=int)[1:-1]

            for i, f in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                detect = face_detector.process(rgb)
                if not detect.detections:
                    continue

                box = detect.detections[0].location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)

                if bw * bh < MIN_FACE_AREA:
                    continue

                pad_w, pad_h = int(bw * PAD_RATIO), int(bh * PAD_RATIO)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = blur_score(gray_crop)
                if sharpness < BLUR_LIMIT:
                    continue

                # Try alignment (optional)
                crop = mesh_align_crop(crop, rgb, (x1, y1, x2, y2))

                try:
                    crop = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue

                name = f"{os.path.splitext(video)[0]}_frame{f}_{i+1}"
                cv2.imwrite(os.path.join(emp_folder, f"{name}.jpg"), crop)
                saved += 1

                if APPLY_AUGMENT:
                    cv2.imwrite(os.path.join(emp_folder, f"{name}_flip.jpg"), cv2.flip(crop, 1))

            cap.release()

    logging.info(f"🖼 {saved} face crops extracted and stored under '{TEMP_IMAGE_DIR}'.")

# ───────────────────────────
# BUILD EMBEDDING EXTRACTOR
# ───────────────────────────
def build_embedding_model():
    base = MobileNetV2(input_shape=FACE_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    model = models.Model(inputs=base.input, outputs=x)
    logging.info("🧠 128-D MobileNetV2 embedding model ready.")
    return model

# ───────────────────────────
# GENERATE EMPLOYEE EMBEDDINGS (AVERAGED)
# ───────────────────────────
def generate_employee_embeddings(model):
    employees = [
        d for d in os.listdir(TEMP_IMAGE_DIR)
        if os.path.isdir(os.path.join(TEMP_IMAGE_DIR, d))
    ]

    vectors = {}
    processed = 0

    for emp_id in employees:
        folder = os.path.join(TEMP_IMAGE_DIR, emp_id)
        faces = []

        for img in os.listdir(folder):
            if not img.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            try:
                arr = img_to_array(load_img(os.path.join(folder, img), target_size=FACE_SIZE)) / 255.0
                emb = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                faces.append(emb)
                processed += 1
            except Exception:
                continue

        if faces:
            vectors[emp_id] = np.mean(np.vstack(faces), axis=0)

    logging.info(f"✅ {len(vectors)} embeddings generated from {processed} images.")
    return vectors

# ───────────────────────────
# SAVE OUTPUT ARTIFACTS
# ───────────────────────────
def save_artifacts(employee_ids, emb_vectors, model):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(CLASS_JSON, "w") as f:
        json.dump({str(i): emp for i, emp in enumerate(employee_ids)}, f, indent=2)

    with open(EMBED_JSON, "w") as f:
        json.dump({k: v.tolist() for k, v in emb_vectors.items()}, f, indent=2)

    model.save(MODEL_SAVE)
    logging.info("🚀 Artifacts saved → ready for TFLite conversion.")

# ───────────────────────────
# CLEANUP TEMP DIRECTORIES
# ───────────────────────────
def cleanup_temp_dirs():
    for d in (TEMP_VIDEO_DIR, TEMP_IMAGE_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
            logging.info(f"🗑 Removed temp directory: {d}")

# ───────────────────────────
# PIPELINE EXECUTION
# ───────────────────────────
if __name__ == "__main__":
    employees = download_videos_from_firebase()
    if not employees:
        raise SystemExit("No videos to process.")

    extract_frames_from_videos(employees)
    model = build_embedding_model()
    embeddings = generate_employee_embeddings(model)

    if not embeddings:
        raise SystemExit("No embeddings created.")

    save_artifacts(employees, embeddings, model)
    cleanup_temp_dirs()
