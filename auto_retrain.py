import os
import cv2
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from firebase_admin import credentials, initialize_app, storage
import base64
import json
import shutil

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Paths & Config
# -----------------------------
USER_VIDEO_DIR = "user_videos_temp"        # Temporary folder for downloaded videos
USER_DATA_DIR = "user_training_data"      # Folder for extracted frames
MODEL_SAVE_PATH = "face_recognition_model.h5"
CLASS_MAPPING_FILE = "class_mapping.json"
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"

FRAME_RATE = 5  # extract 1 frame every N frames

# -----------------------------
# Firebase Initialization
# -----------------------------
def initialize_firebase():
    service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
    if not service_account_base64:
        logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found.")
        raise ValueError("Firebase Service Account Key missing.")

    decoded_json = base64.b64decode(service_account_base64).decode('utf-8')
    cred = credentials.Certificate(json.loads(decoded_json))
    initialize_app(cred, {'storageBucket': FIREBASE_BUCKET_NAME})
    logging.info("Firebase initialized.")

# -----------------------------
# Download videos
# -----------------------------
def download_videos():
    initialize_firebase()
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="video_training_data/")  # list all videos
    os.makedirs(USER_VIDEO_DIR, exist_ok=True)

    employee_ids = set()
    for blob in blobs:
        if blob.name.endswith(".mp4"):
            employee_id = blob.name.split("/")[1]
            employee_ids.add(employee_id)
            dest_dir = os.path.join(USER_VIDEO_DIR, employee_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(blob.name))
            blob.download_to_filename(dest_path)
            logging.info(f"Downloaded {blob.name} to {dest_path}")
    return list(employee_ids)

# -----------------------------
# Extract frames
# -----------------------------
def extract_frames(employee_ids):
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    total_frames = 0
    for emp_id in employee_ids:
        video_folder = os.path.join(USER_VIDEO_DIR, emp_id)
        output_folder = os.path.join(USER_DATA_DIR, emp_id)
        os.makedirs(output_folder, exist_ok=True)

        for video_file in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % FRAME_RATE == 0:
                    frame_name = os.path.join(output_folder, f"{video_file}_{saved_count}.jpg")
                    cv2.imwrite(frame_name, frame)
                    saved_count += 1
                    total_frames += 1
                frame_count += 1
            cap.release()
            logging.info(f"Extracted {saved_count} frames from {video_path}")
    logging.info(f"Total frames extracted: {total_frames}")

# -----------------------------
# Build & Train Model
# -----------------------------
def build_and_train_model():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        USER_DATA_DIR,
        target_size=(160, 160),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        USER_DATA_DIR,
        target_size=(160, 160),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160,160,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info(f"Starting training on {len(train_gen.class_indices)} classes...")
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Model saved at {MODEL_SAVE_PATH}")

    # Save class mapping
    with open(CLASS_MAPPING_FILE, 'w') as f:
        json.dump(train_gen.class_indices, f)
    logging.info(f"Class mapping saved at {CLASS_MAPPING_FILE}")

# -----------------------------
# Cleanup
# -----------------------------
def cleanup():
    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
        logging.info(f"Temporary video folder {USER_VIDEO_DIR} removed.")
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
        logging.info(f"Training image folder {USER_DATA_DIR} removed.")

# -----------------------------
# Main Flow
# -----------------------------
if __name__ == "__main__":
    emp_ids = download_videos()
    extract_frames(emp_ids)
    build_and_train_model()
    cleanup()
