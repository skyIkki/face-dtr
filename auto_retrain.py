#!/usr/bin/env python3
"""
auto_retrain.py (Face-Cropped, Quality-Filtered Version)

- Downloads videos from Firebase.
- Extracts only face crops using MediaPipe Face Detection.
- Adds padding, optional alignment, blur filtering.
- Resizes crops to TARGET_SIZE for consistent embeddings.
- Builds MobileNetV2 embedding extractor (128-D normalized).
- Averages embeddings per employee and saves artifacts in mobile_artifacts/..
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

# MediaPipe face detection
import mediapipe as mp
mp_face = mp.solutions.face_detection
# We'll instantiate detector once and reuse it
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

TARGET_SIZE = (160, 160)      # final saved face image size (w,h)
NUM_SAMPLES_PER_VIDEO = 10    # frames sampled per video
PAD_RATIO = 0.35              # padding around detected bbox (35%)
BLUR_THRESHOLD = 40.0         # Laplacian variance threshold; lower = blurrier
AUGMENTATIONS = True          # if True, also save a horizontally flipped crop for augmentation

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
        # only video file blobs
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
# Helpers: alignment, blur check
# -----------------------------
def rotate_image(img, angle, center=None):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def compute_blur_score(gray_img):
    """Return variance of Laplacian (higher => sharper)."""
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def try_align_face_using_eyes(crop_rgb, detection):
    """
    Attempt to align face crop so eyes are horizontal.
    detection: MediaPipe detection object with relative_keypoints if available.
    Returns rotated crop_rgb (or original if cannot align).
    """
    try:
        keypoints = detection.location_data.relative_keypoints
        # MediaPipe face_detection relative_keypoints ordering: (RIGHT_EYE, LEFT_EYE, NOSE, MOUTH_CENTER, RIGHT_EAR, LEFT_EAR) in some versions.
        # We'll try to read first two as eyes; if not available, abort safely.
        if len(keypoints) < 2:
            return crop_rgb
        # compute eye coords in crop image coordinates: relative_keypoints are relative to image, but we will compute angle from them
        # However detection.location_data.relative_bounding_box gives bbox; eyes coords are relative to full image.
        # For safety we'll not rely on mapping; instead if keys present, use them relative to crop by re-scaling below where called.
        return crop_rgb
    except Exception:
        return crop_rgb

# -----------------------------
# FACE-ONLY Extractor (improved)
# -----------------------------
def extract_frames_from_videos(employee_ids):
    """
    Extract high-quality face-only crops with padding + optional alignment + blur filtering.
    Saves crops resized to TARGET_SIZE under USER_DATA_DIR/<employee_id>/
    """
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    total_saved = 0
    for emp_id in employee_ids:
        vid_dir = os.path.join(USER_VIDEO_DIR, emp_id)
        out_dir = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(out_dir, exist_ok=True)

        logging.info(f"Processing employee: {emp_id} (videos in {vid_dir})")
        if not os.path.exists(vid_dir):
            logging.warning(f"No video directory found for {emp_id}, skipping.")
            continue

        for video_file in sorted(os.listdir(vid_dir)):
            if not video_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            video_path = os.path.join(vid_dir, video_file)
            logging.info(f"  Video: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.warning(f"    Could not open video {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if frame_count == 0:
                logging.warning(f"    No frames in {video_path}")
                cap.release()
                continue

            indices = np.linspace(0, frame_count - 1, NUM_SAMPLES_PER_VIDEO + 2, dtype=int)[1:-1]

            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # convert to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detector.process(rgb)

                if not result.detections:
                    logging.debug(f"    Frame {idx}: no face detected")
                    continue

                # take first / largest detection - MediaPipe returns detections ordered by score
                det = result.detections[0]
                rel_bbox = det.location_data.relative_bounding_box

                h, w, _ = frame.shape
                # compute bbox in pixel coords
                x = int(rel_bbox.xmin * w)
                y = int(rel_bbox.ymin * h)
                bw = int(rel_bbox.width * w)
                bh = int(rel_bbox.height * h)

                # padding
                pad_x = int(bw * PAD_RATIO)
                pad_y = int(bh * PAD_RATIO)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)

                # ensure valid
                if x2 <= x1 or y2 <= y1:
                    logging.debug(f"    Frame {idx}: invalid bbox after padding, skip")
                    continue

                crop = frame[y1:y2, x1:x2]  # BGR crop
                if crop is None or crop.size == 0:
                    continue

                # QUICK BLUR CHECK
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_score = compute_blur_score(gray_crop)
                if blur_score < BLUR_THRESHOLD:
                    logging.debug(f"    Frame {idx}: blur_score={blur_score:.2f} < {BLUR_THRESHOLD}, skipping.")
                    continue

                # OPTIONAL: attempt to align face using eye keypoints.
                # MediaPipe detection provides keypoints in detection.location_data.relative_keypoints,
                # but mapping them precisely to the crop coordinates is tricky—so we'll attempt a robust method:
                try:
                    # get full-image keypoint coords if present
                    rk = det.location_data.relative_keypoints
                    if rk and len(rk) >= 2:
                        # take two first points as eyes (this works for many versions)
                        left_eye_rel = rk[0]  # x,y relative to image
                        right_eye_rel = rk[1]
                        # convert to absolute image coords
                        left_eye_x = int(left_eye_rel.x * w)
                        left_eye_y = int(left_eye_rel.y * h)
                        right_eye_x = int(right_eye_rel.x * w)
                        right_eye_y = int(right_eye_rel.y * h)

                        # convert to crop-relative coords
                        le_x_c = left_eye_x - x1
                        le_y_c = left_eye_y - y1
                        re_x_c = right_eye_x - x1
                        re_y_c = right_eye_y - y1

                        # compute angle (degrees) between eyes
                        dx = re_x_c - le_x_c
                        dy = re_y_c - le_y_c
                        if abs(dx) > 1e-3:
                            angle = math.degrees(math.atan2(dy, dx))
                            # rotate around center of crop
                            crop = rotate_image(crop, angle, center=((crop.shape[1]//2), (crop.shape[0]//2)))
                            # recompute gray for blur check after rotate (optional)
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            blur_score = compute_blur_score(gray_crop)
                            if blur_score < BLUR_THRESHOLD:
                                logging.debug(f"    Frame {idx}: post-rotation blur {blur_score:.2f}, skipping.")
                                continue
                except Exception as e:
                    # alignment best-effort only; on failure continue with original crop
                    logging.debug(f"    Frame {idx}: alignment attempt failed: {e}")

                # Resize to TARGET_SIZE (width, height)
                try:
                    resized = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    logging.debug(f"    Frame {idx}: resize error: {e}")
                    continue

                # Save main crop
                base_name = f"{os.path.splitext(video_file)[0]}_f{idx}_face_{i+1}"
                out_path = os.path.join(out_dir, f"{base_name}.jpg")
                try:
                    cv2.imwrite(out_path, resized)
                    total_saved += 1
                except Exception as e:
                    logging.warning(f"    Could not write {out_path}: {e}")
                    continue

                # augmentation: horizontal flip (optional)
                if AUGMENTATIONS:
                    try:
                        flipped = cv2.flip(resized, 1)
                        out_path_f = os.path.join(out_dir, f"{base_name}_flip.jpg")
                        cv2.imwrite(out_path_f, flipped)
                        total_saved += 1
                    except Exception:
                        pass

                logging.debug(f"    Saved crop {out_path} (blur={blur_score:.2f})")

            cap.release()

    logging.info(f"Extracted and saved {total_saved} face crops total under '{USER_DATA_DIR}'.")

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
    logging.info("Built MobileNetV2 embedding model.")
    return model

# -----------------------------
# Embeddings generation
# -----------------------------
def generate_embeddings(model):
    emp_dirs = [d for d in os.listdir(USER_DATA_DIR) if os.path.isdir(os.path.join(USER_DATA_DIR, d))]
    embeddings = {}
    total_images = 0

    for emp_id in emp_dirs:
        folder = os.path.join(USER_DATA_DIR, emp_id)
        vectors = []
        for img_file in os.listdir(folder):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, img_file)
            try:
                img = load_img(img_path, target_size=TARGET_SIZE)
                arr = img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                emb = model.predict(arr, verbose=0)[0]
                vectors.append(emb)
                total_images += 1
            except Exception as e:
                logging.debug(f"Failed to process {img_path}: {e}")

        if len(vectors) > 0:
            embeddings[emp_id] = np.mean(np.vstack(vectors), axis=0)
            logging.info(f"Generated embedding for {emp_id} from {len(vectors)} images.")
        else:
            logging.warning(f"No valid face images found for {emp_id}; skipping embedding.")

    logging.info(f"Processed {total_images} images to generate embeddings for {len(embeddings)} employees.")
    return embeddings

# -----------------------------
# Save artifacts
# -----------------------------
def save_class_mapping(employee_ids):
    mapping = {str(i): emp for i, emp in enumerate(employee_ids)}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
    logging.info(f"Saved class mapping to {CLASS_MAPPING_FILE}")

def save_embeddings_json(emb_dict):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({k: v.tolist() for k, v in emb_dict.items()}, f, indent=4)
    logging.info(f"Saved embeddings JSON to {EMBEDDINGS_FILE}")

# -----------------------------
# Cleanup
# -----------------------------
def cleanup_temp_dirs():
    for d in (USER_VIDEO_DIR, USER_DATA_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
            logging.info(f"Removed temp dir {d}")

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

        model = build_embedding_model()
        embeddings = generate_embeddings(model)

        if not embeddings:
            logging.critical("No embeddings generated; aborting save.")
            raise SystemExit(1)

        save_embeddings_json(embeddings)
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Saved embedding model to {MODEL_SAVE_PATH}. Artifacts ready for convert_to_tflite.py")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise

    finally:
        cleanup_temp_dirs()
