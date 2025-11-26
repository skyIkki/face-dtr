#!/usr/bin/env python3
import os
import json
import logging
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, BatchNormalization, Activation, ReLU
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2 # <-- IMPORTED
from moviepy.editor import VideoFileClip
from sklearn.cluster import DBSCAN

# --- CONFIGURATION ---
DATA_DIR = "video_training_data"
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
MODEL_FILENAME = "face_embedding_model.h5"
EMBEDDINGS_JSON_FILENAME = "employee_embeddings.json"

# Face Extraction/Quality Params
INPUT_SIZE = 160 # Target size for face crops (must match Java TFLite)
PAD_RATIO = 0.35 # Padding around bounding box (must match Java inference)
MIN_FACE_AREA_RATIO = 0.005 # Minimum face area relative to frame
BLUR_THRESHOLD = 15.0 # Lower is sharper
NUM_SAMPLES_PER_VIDEO = 30 # Increased for better coverage of instructed poses

# --- INITIAL SETUP ---
os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODEL ARCHITECTURE ---
def create_embedding_model(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), embedding_dim=128):
    # Base: MobileNetV2 without the top classification layer
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base weights

    # Custom top layers for embedding
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim, use_bias=False)(x)
    # L2 Normalization is CRUCIAL for distance/similarity calculations
    x = tf.math.l2_normalize(x, axis=1)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# --- FACE EXTRACTION & QUALITY ---
def extract_and_align_face(frame, face_landmarks_list):
    # Implementation using MediaPipe FaceMesh to align, crop, and pad
    # (Assuming your original detailed implementation is here)
    # Ensure the returned crop uses the PAD_RATIO and INPUT_SIZE
    # ... Your existing alignment/crop logic ...

    # PLACEHOLDER for your existing extraction logic
    # The crucial part is the padding and resizing:
    # 1. Get face bounding box (via ML Kit/MediaPipe)
    # 2. Apply PAD_RATIO to the box
    # 3. Crop the frame to the padded box
    # 4. Resize the cropped face to (INPUT_SIZE, INPUT_SIZE) e.g., 160x160
    # 5. Return the PIL/OpenCV image
    
    # Since this is a massive block, I'll focus on the data handling:
    return frame # Placeholder - must be an aligned, padded, and resized image (INPUT_SIZE x INPUT_SIZE)

def check_blur(image):
    # Calculates the Laplacian variance to estimate blur
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def generate_embeddings(video_path, model):
    embeddings = []
    
    try:
        clip = VideoFileClip(video_path)
        fps = clip.fps
        duration = clip.duration
        
        # Sample frames linearly across the video
        num_frames_to_sample = NUM_SAMPLES_PER_VIDEO
        frame_indices = np.linspace(0, duration * fps - 1, num_frames_to_sample, dtype=int)
        
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            for i, frame_index in enumerate(frame_indices):
                try:
                    frame = clip.get_frame(frame_index / fps)
                    
                    # Convert BGR (OpenCV standard from moviepy) to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        # Extract, align, pad, and resize the face
                        face_img = extract_and_align_face(rgb_frame, results.multi_face_landmarks[0])
                        
                        # Check quality
                        blur_score = check_blur(face_img)
                        if blur_score < BLUR_THRESHOLD:
                            logging.warning(f"Frame {i+1} skipped: Too blurry (Score: {blur_score:.2f})")
                            continue

                        # 1. Convert to array
                        arr = img_to_array(face_img)

                        # 2. CORRECT NORMALIZATION: Scale to [-1.0, 1.0] for MobileNetV2
                        arr = preprocess_input(arr)
                        
                        # 3. Predict
                        arr = np.expand_dims(arr, axis=0)
                        emb = model.predict(arr, verbose=0)[0]
                        embeddings.append(emb)

                except Exception as e:
                    logging.error(f"Error processing frame {frame_index}: {e}")
                    continue
        
    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return None

    if not embeddings:
        logging.warning(f"No valid embeddings generated for {video_path}.")
        return None
    
    # Average the high-quality embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding.tolist()

# --- MAIN EXECUTION ---
def auto_retrain_pipeline():
    logging.info("Starting Auto-Retrain Pipeline...")
    
    # 1. Load/Create Model
    model = create_embedding_model()
    
    # 2. Collect employee video data
    employee_embeddings = {}
    employee_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not employee_dirs:
        logging.warning("No employee video data found. Skipping embedding generation.")
        return

    # 3. Generate Embeddings
    for employee_id in employee_dirs:
        employee_path = os.path.join(DATA_DIR, employee_id)
        video_files = [os.path.join(employee_path, f) for f in os.listdir(employee_path) if f.endswith('.mp4')]
        
        if not video_files:
            logging.warning(f"No videos found for employee {employee_id}. Skipping.")
            continue
            
        logging.info(f"Processing {len(video_files)} video(s) for Employee: {employee_id}")
        
        all_embeddings = []
        for video_file in video_files:
            embedding = generate_embeddings(video_file, model)
            if embedding:
                all_embeddings.append(embedding)

        if all_embeddings:
            # Average embeddings across all videos for this employee
            final_embedding = np.mean(all_embeddings, axis=0).tolist()
            employee_embeddings[employee_id] = final_embedding
            logging.info(f"Generated final embedding for {employee_id}.")
            
    # 4. Save Artifacts
    if employee_embeddings:
        # Save Keras model (.h5)
        model.save(os.path.join(MOBILE_ARTIFACTS_DIR, MODEL_FILENAME))
        logging.info(f"✅ Saved Keras model to {MOBILE_ARTIFACTS_DIR}/{MODEL_FILENAME}")

        # Save embeddings JSON
        with open(os.path.join(MOBILE_ARTIFACTS_DIR, EMBEDDINGS_JSON_FILENAME), 'w') as f:
            json.dump(employee_embeddings, f, indent=4)
        logging.info(f"✅ Saved employee embeddings to {MOBILE_ARTIFACTS_DIR}/{EMBEDDINGS_JSON_FILENAME}")
    else:
        logging.warning("No successful embeddings generated. Artifacts not updated.")

    # Save class mapping (employee ID list for Java)
    class_mapping = {idx: name for idx, name in enumerate(employee_embeddings.keys())}
    with open(os.path.join(MOBILE_ARTIFACTS_DIR, "class_mapping.json"), 'w') as f:
        json.dump(class_mapping, f, indent=4)
    logging.info(f"✅ Saved class mapping.")

if __name__ == "__main__":
    auto_retrain_pipeline()
