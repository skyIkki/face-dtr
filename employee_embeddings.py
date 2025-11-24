# employee_embeddings.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
MOBILE_ARTIFACTS_DIR = "mobile_artifacts"
MODEL_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_recognition_model.tflite")  # Optional if using H5 embeddings
CLASS_MAPPING_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "class_mapping.json")
EMBEDDINGS_OUTPUT_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "employee_embeddings.json")
TRAINING_DATA_DIR = "training_data"  # Your folders with images per employee ID

# Load class mapping
with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)  # { "0": "2019-0001", "1": "2019-0002", ... }

# If you want embeddings from Keras model (H5), load the model
MODEL_H5_PATH = os.path.join(MOBILE_ARTIFACTS_DIR, "face_recognition_model.h5")
from tensorflow.keras.models import load_model
model = load_model(MODEL_H5_PATH)

def preprocess_img(img_path, target_size=(160, 160)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

employee_embeddings = {}

for class_index, employee_id in class_mapping.items():
    class_folder = os.path.join(TRAINING_DATA_DIR, employee_id)
    if not os.path.exists(class_folder):
        print(f"⚠️ Training folder missing for {employee_id}")
        continue

    embeddings_list = []
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        try:
            img_array = preprocess_img(img_path)
            embedding = model.predict(img_array)[0]  # Assuming model outputs embeddings
            embeddings_list.append(embedding.tolist())
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    if embeddings_list:
        # Average embedding per employee
        avg_embedding = np.mean(np.array(embeddings_list), axis=0)
        employee_embeddings[employee_id] = avg_embedding.tolist()

# Save to JSON
os.makedirs(MOBILE_ARTIFACTS_DIR, exist_ok=True)
with open(EMBEDDINGS_OUTPUT_PATH, "w") as f:
    json.dump(employee_embeddings, f)

print(f"✅ Saved employee embeddings to {EMBEDDINGS_OUTPUT_PATH}")
