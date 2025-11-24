# employee_embeddings.py
import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import FaceRecognitionModel  # Or replace with your own model import

# --- CONFIG ---
TRAINING_DATA_DIR = "training_data"
OUTPUT_DIR = "mobile_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODEL SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FaceRecognitionModel()  # Replace with your model class
model.load_state_dict(torch.load("face_recognition_model.pth", map_location=device))  # your trained PyTorch weights
model.eval()
model.to(device)

# --- TRANSFORM ---
input_size = 160
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1] normalization
])

# --- EMBEDDING EXTRACTION ---
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x)
    return embedding.squeeze().cpu().numpy()

# --- MAIN PROCESS ---
embeddings_dict = {}
for employee_id in os.listdir(TRAINING_DATA_DIR):
    employee_path = os.path.join(TRAINING_DATA_DIR, employee_id)
    if not os.path.isdir(employee_path):
        continue

    all_embeddings = []
    for img_file in os.listdir(employee_path):
        img_path = os.path.join(employee_path, img_file)
        try:
            emb = get_embedding(img_path)
            all_embeddings.append(emb)
        except Exception as e:
            print(f"❌ Failed for {img_path}: {e}")

    if all_embeddings:
        avg_embedding = np.mean(all_embeddings, axis=0)
        embeddings_dict[employee_id] = avg_embedding

        # Save JSON for Android
        out_file = os.path.join(OUTPUT_DIR, f"{employee_id}.json")
        with open(out_file, "w") as f:
            json.dump({"employeeID": employee_id, "embedding": avg_embedding.tolist()}, f)
        print(f"✅ Saved embedding for {employee_id}")

print("🎉 All embeddings generated successfully!")
