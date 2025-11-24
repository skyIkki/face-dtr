# generate_embeddings.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
MODEL_PATH = "mobile_artifacts/face_recognition_model.tflite"  # Or h5 if using Keras
CLASS_MAPPING_PATH = "mobile_artifacts/class_mapping.json"
OUTPUT_PATH = "mobile_artifacts/employee_embeddings.json"
TRAINING_DATA_PATH = "training_data/"  # Same structure as before

# Load class mapping
with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)  # e.g., {"2019-0001":0, "2019-0002":1}

# Load your Keras model (if using .h5)
from tensorflow.keras.models import load_model
model = load_model("mobile_artifacts/face_recognition_model.h5")

def get_embedding(img_path):
    """Load image and return normalized embedding vector from model"""
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    img_array = img_array / 255.0  # Normalize
    embedding = model.predict(img_array)
    return embedding[0]  # Return as 1D array

# Generate embeddings per employee
employee_embeddings = {}

for employee_id in class_mapping:
    emp_folder = os.path.join(TRAINING_DATA_PATH, employee_id)
    if not os.path.exists(emp_folder):
        continue

    embeddings_list = []
    for img_file in os.listdir(emp_folder):
        img_path = os.path.join(emp_folder, img_file)
        embedding = get_embedding(img_path)
        embeddings_list.append(embedding)

    # Average embedding per employee
    if embeddings_list:
        avg_embedding = np.mean(embeddings_list, axis=0).tolist()
        employee_embeddings[employee_id] = avg_embedding

# Save embeddings to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(employee_embeddings, f)

print(f"✅ Generated employee_embeddings.json at {OUTPUT_PATH}")
