import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
import firebase_admin
from firebase_admin import credentials, storage as fb_storage
import logging
import json
import shutil
import base64

# -----------------------------
# Config & Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Directory Configuration
# NOTE: BASE_DATA_DIR will be the unified directory for ALL training data (base + user)
BASE_DATA_DIR = "training_data_unified" 
USER_VIDEO_DIR = "user_videos_temp" # Temporary folder for downloaded raw videos
MODEL_SAVE_PATH = "face_recognition_model.h5"
CLASS_MAPPING_FILE = "class_mapping.json"

# Firebase Storage Configuration
FIREBASE_BUCKET_NAME = "face-dtr-6efa3.firebasestorage.app"
FIREBASE_USER_DATA_PREFIX = "video_training_data/" # Assumes structure like video_training_data/CLASS_NAME/video.mp4

# Training Parameters
TARGET_SIZE = (160, 160)
BATCH_SIZE = 32
NUM_EPOCHS = 10
VALIDATION_SPLIT = 0.2

# -----------------------------
# Firebase Setup
# -----------------------------
def initialize_firebase():
    """Initializes Firebase Admin SDK using a Base64-encoded service account key from an environment variable."""
    if firebase_admin._apps:
        logging.info("Firebase already initialized.")
        return

    logging.info("Initializing Firebase Admin SDK...")
    try:
        # Load credentials from environment variable (safer for CI/CD)
        service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
        if not service_account_base64:
            logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found. Cannot initialize Firebase.")
            raise ValueError("Firebase Service Account Key missing.")

        service_account_json_decoded = base64.b64decode(service_account_base64).decode('utf-8')
        
        cred = credentials.Certificate(json.loads(service_account_json_decoded))
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': FIREBASE_BUCKET_NAME
        })
        logging.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Firebase Admin SDK: {e}")
        raise

# -----------------------------
# Download Videos from Firebase (Maintaining Class Structure)
# -----------------------------
def download_user_data_from_firebase():
    """Downloads user videos, ensuring the local directory structure mirrors the class structure in Firebase."""
    initialize_firebase()
    bucket = fb_storage.bucket()
    
    # Ensure temporary video directory is clean
    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
    os.makedirs(USER_VIDEO_DIR, exist_ok=True)

    blobs = bucket.list_blobs(prefix=FIREBASE_USER_DATA_PREFIX)
    downloaded_videos_count = 0
    extracted_classes = set()

    for blob in blobs:
        # Skip directories and non-video files
        if blob.name.endswith('/') or not blob.name.lower().endswith(('.mp4', '.mov', '.avi')):
            continue

        # Path parsing logic: FIREBASE_USER_DATA_PREFIX/CLASS_NAME/video.mp4
        relative_path = blob.name[len(FIREBASE_USER_DATA_PREFIX):]
        parts = relative_path.split('/')
        
        if len(parts) < 2:
            logging.warning(f"Skipping blob with unusual structure: {blob.name}")
            continue

        class_name = parts[0]
        video_filename = parts[-1]
        
        # Create local path structure: user_videos_temp/CLASS_NAME/video.mp4
        local_class_dir = os.path.join(USER_VIDEO_DIR, class_name)
        os.makedirs(local_class_dir, exist_ok=True)
        local_path = os.path.join(local_class_dir, video_filename)
        
        try:
            blob.download_to_filename(local_path)
            downloaded_videos_count += 1
            extracted_classes.add(class_name)
            logging.debug(f"Downloaded: {blob.name} to {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")

    logging.info(f"✅ Downloaded {downloaded_videos_count} videos for {len(extracted_classes)} classes.")
    return extracted_classes


# -----------------------------
# Extract Frames and Merge into BASE_DATA_DIR (Maintaining Class Structure)
# -----------------------------
def extract_frames_and_merge(class_names):
    """Processes videos, extracts 5 uniformly sampled frames, and saves them directly into the BASE_DATA_DIR."""
    total_frames = 0
    videos_processed = 0

    # Ensure BASE_DATA_DIR exists for all classes
    for class_name in class_names:
        os.makedirs(os.path.join(BASE_DATA_DIR, class_name), exist_ok=True)

    for class_name in os.listdir(USER_VIDEO_DIR):
        video_class_dir = os.path.join(USER_VIDEO_DIR, class_name)
        if not os.path.isdir(video_class_dir):
            continue

        output_class_dir = os.path.join(BASE_DATA_DIR, class_name)
        
        for video_filename in os.listdir(video_class_dir):
            if not video_filename.lower().endswith(('.mp4', '.mov', '.avi')):
                continue

            video_path = os.path.join(video_class_dir, video_filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.warning(f"Could not open video file: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = frame_count / fps if fps > 0 else 0
            
            NUM_SAMPLES = 5 # Extract 5 frames per video
            
            if duration_sec > 0:
                # Calculate time points (in seconds) for uniform sampling
                time_points = np.linspace(0, duration_sec, NUM_SAMPLES + 2)[1:-1]
            else:
                logging.warning(f"Video {video_path} has zero duration. Skipping.")
                cap.release()
                continue
            
            video_name_base = os.path.splitext(video_filename)[0]
            
            for i, t in enumerate(time_points):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                
                if ret:
                    frame_output_path = os.path.join(output_class_dir, f"{video_name_base}_frame_{i+1}.jpg")
                    cv2.imwrite(frame_output_path, frame)
                    total_frames += 1
            
            cap.release()
            videos_processed += 1

    logging.info(f"✅ Finished frame extraction. Processed {videos_processed} videos and extracted {total_frames} frames.")
    
    # Cleanup temporary video files
    if os.path.exists(USER_VIDEO_DIR):
        shutil.rmtree(USER_VIDEO_DIR)
        logging.info(f"Cleaned up temporary video directory: {USER_VIDEO_DIR}")

# -----------------------------
# Load Images for Training (Unified Approach)
# -----------------------------
def load_unified_dataset():
    """Loads all images (base + user frames) from the unified BASE_DATA_DIR."""
    
    # Check if BASE_DATA_DIR has any data at all
    if not os.path.exists(BASE_DATA_DIR) or not os.listdir(BASE_DATA_DIR):
        logging.critical(f"Unified data directory '{BASE_DATA_DIR}' is empty. Cannot train model.")
        raise ValueError("No data available for training.")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    # Use the same generator instance for both train and validation splits on the unified data
    train_gen = datagen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        BASE_DATA_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False # Typically false for validation
    )
    
    logging.info(f"Loaded {train_gen.samples} training images and {val_gen.samples} validation images.")
    logging.info(f"Total unique classes: {train_gen.num_classes}")
    
    # Save the class mapping for inference later
    class_mapping = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(class_mapping, f, indent=4)
        
    return train_gen, val_gen, train_gen.num_classes

# -----------------------------
# Build Model (using Transfer Learning for better performance)
# -----------------------------
def build_model(num_classes):
    """Builds a model using MobileNetV2 pre-trained base for better feature extraction."""
    
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=TARGET_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the layers of the base model
    base_model.trainable = False

    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info(f"Built new MobileNetV2-based model with {num_classes} output classes.")
    return model

# -----------------------------
# Upload Model and Mapping to Firebase Storage
# -----------------------------
def upload_artifacts_to_firebase():
    """Uploads the trained model and class mapping to Firebase Storage."""
    initialize_firebase()
    bucket = fb_storage.bucket()

    # Model upload
    model_blob = bucket.blob(MODEL_SAVE_PATH)
    model_blob.upload_from_filename(MODEL_SAVE_PATH)
    logging.info(f"✅ Uploaded Model: {MODEL_SAVE_PATH}")

    # Mapping upload
    mapping_blob = bucket.blob(CLASS_MAPPING_FILE)
    mapping_blob.upload_from_filename(CLASS_MAPPING_FILE)
    logging.info(f"✅ Uploaded Class Mapping: {CLASS_MAPPING_FILE}")
    
    # Increment model version (optional, but good practice)
    # This logic assumes the version file sits flat in the bucket root for simplicity
    model_version_file = "model_version.txt"
    remote_version_path = model_version_file
    
    current_version = 0
    try:
        version_blob = bucket.blob(remote_version_path)
        version_data = version_blob.download_as_text()
        current_version = int(version_data.strip())
    except Exception:
        logging.warning("Could not retrieve existing model version. Starting at 0.")

    new_version = current_version + 1
    
    with open(model_version_file, "w") as f:
        f.write(str(new_version))

    version_blob.upload_from_filename(model_version_file)
    logging.info(f"✅ Incremented and uploaded model version to v{new_version}")


# -----------------------------
# Main Training Flow
# -----------------------------
def main():
    try:
        # 1️⃣ Download videos from Firebase (maintaining class structure)
        # This will create user_videos_temp/CLASS_NAME/video.mp4
        user_classes = download_user_data_from_firebase()
        
        # 2️⃣ Extract frames and merge them into the BASE_DATA_DIR
        # This moves user data into training_data_unified/CLASS_NAME/frame.jpg
        # Assumes base data is already in training_data_unified/BASE_CLASS/image.jpg
        extract_frames_and_merge(user_classes)

        # 3️⃣ Load unified datasets
        train_gen, val_gen, num_classes = load_unified_dataset()
        
        # Guard against zero data
        if train_gen.samples == 0:
            logging.critical("Combined training dataset is empty after merging. Aborting.")
            return

        # 4️⃣ Build / load model
        if os.path.exists(MODEL_SAVE_PATH):
            model = load_model(MODEL_SAVE_PATH)
            logging.info(f"✅ Loaded existing model from {MODEL_SAVE_PATH}")
            # Ensure the output layer size is correct (optional check)
            if model.layers[-1].output_shape[1] != num_classes:
                 logging.warning(f"Existing model output shape ({model.layers[-1].output_shape[1]}) does not match new classes ({num_classes}). Rebuilding model.")
                 model = build_model(num_classes)
        else:
            model = build_model(num_classes)
            logging.info("✅ Built new MobileNetV2-based model")

        # 5️⃣ Train model
        logging.info("🚀 Starting training on unified dataset...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=NUM_EPOCHS,
            verbose=2 # Show progress
        )
        logging.info("✅ Training complete.")

        # 6️⃣ Save model locally
        model.save(MODEL_SAVE_PATH)
        logging.info(f"✅ Model saved locally to {MODEL_SAVE_PATH}")
        logging.info(f"✅ MODEL SAVED AT: {os.path.abspath("face_recognition_model.h5")}")
        # 7️⃣ Upload model and artifacts to Firebase
        upload_artifacts_to_firebase()
        
        # 8️⃣ Final Cleanup (remove all local artifacts)
        logging.info("Cleaning up local files...")
        if os.path.exists(BASE_DATA_DIR):
            shutil.rmtree(BASE_DATA_DIR)
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
        if os.path.exists(CLASS_MAPPING_FILE):
            os.remove(CLASS_MAPPING_FILE)
        if os.path.exists("model_version.txt"):
            os.remove("model_version.txt")
        logging.info("Cleanup successful. Pipeline finished.")

    except Exception as e:
        logging.critical(f"An error occurred in the main pipeline: {e}")
        # Ensure temporary files are cleaned up even on failure
        if os.path.exists(USER_VIDEO_DIR):
            shutil.rmtree(USER_VIDEO_DIR)

if __name__ == "__main__":
    main()
