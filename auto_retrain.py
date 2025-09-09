import os
import io
import zipfile
import urllib.request
import tarfile
import shutil
import scipy.io
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
import requests
import json
import logging
import random
import PIL.Image # Ensure PIL is imported for the UnifiedDataset

# --- NEW IMPORTS FOR FIREBASE STORAGE ---
import firebase_admin
from firebase_admin import credentials, storage
import base64 # <--- THIS IS THE NEW IMPORT
# ----------------------------------------

# --- DEBUGGING LINES AT THE VERY TOP ---
print("DEBUG: auto_retrain.py script execution started.")
print(f"DEBUG: Current working directory: {os.getcwd()}")
# -------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("DEBUG: Logging configured successfully.")

# Set random seeds for reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
set_all_seeds(42)

logging.info("DEBUG: Random seeds set.")

# ---------------------------
# CONFIGURATION
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUTPUT = "best_flower_model_v3.pt"
CLASS_MAPPING_FILE = "class_to_label.json"
# REMOVED: DOWNLOAD_URL = "https://flower-upload-api.onrender.com/download-data"
USER_DATA_DIR = "user_training_data"
BASE_TRAINING_DATA_DIR = "base_training_data"

# NEW: Firebase Storage Bucket Name (CHECK THIS IN YOUR FIREBASE CONSOLE -> STORAGE)
# It's usually something like 'your-project-id.appspot.com'
FIREBASE_STORAGE_BUCKET = "flower-identification-c2ef6.firebasestorage.app"

# NEW: Path within Firebase Storage where user data is uploaded by the Android app
FIREBASE_USER_DATA_PREFIX = "user_training_data/"

# NEW: Path within Firebase Storage where the trained model and mapping will be uploaded
FIREBASE_MODEL_UPLOAD_PREFIX = "" # Upload to root of bucket, or 'models/' if you prefer a subfolder

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
VALIDATION_SPLIT_RATIO = 0.2 # 20% of combined data for validation

# Oxford 102 Flowers dataset URLs (still not used for training data)
# These are kept for context but are not actively used for data loading anymore
FLOWER_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
SETID_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# Oxford mapping (still not used for training data classes) - kept for completeness but not actively used for training class names
OXFORD_CODE_TO_NAME_MAP = {
    "001": "pink primrose", "002": "globe thistle", "003": "blanket flower", "004": "trumpet creeper",
    "005": "blackberry lily", "006": "snapdragon", "007": "colt's foot", "008": "king protea",
    "009": "spear thistle", "010": "yellow iris", "011": "globe-flower", "012": "purple coneflower",
    "013": "peruvian lily", "014": "balloon flower", "015": "hard-leaved pocket orchid",
    "016": "giant white arum lily", "017": "fire lily", "018": "pincushion flower", "019": "fritillary",
    "020": "red ginger", "021": "grape hyacinth", "022": "corn poppy", "023": "prince of wales feathers",
    "024": "stemless gentian", "025": "artichoke", "026": "canterbury bells", "027": "sweet william",
    "028": "carnation", "029": "garden phlox", "030": "love in the mist", "031": "mexican aster",
    "032": "alpine sea holly", "033": "ruby-lipped cattleya", "034": "cape flower", "035": "great masterwort",
    "036": "siam tulip", "037": "sweet pea", "038": "lenten rose", "039": "barberton daisy",
    "040": "daffodil", "041": "sword lily", "042": "poinsettia", "043": "bolero deep blue",
    "044": "wallflower", "045": "marigold", "046": "buttercup", "047": "oxeye daisy",
    "048": "english marigold", "049": "common dandelion", "050": "petunia", "051": "wild pansy",
    "052": "primula", "053": "sunflower", "054": "pelargonium", "055": "bishop of llandaff",
    "056": "gaura", "057": "geranium", "058": "orange dahlia", "059": "tiger lily",
    "060": "pink-yellow dahlia", "061": "cautleya spicata", "062": "japanese anemone",
    "063": "black-eyed susan", "064": "silverbush", "065": "californian poppy", "066": "osteospermum",
    "067": "spring crocus", "068": "bearded iris", "069": "windflower", "070": "moon orchid",
    "071": "tree poppy", "072": "gazania", "073": "azalea", "074": "water lily",
    "075": "rose", "076": "thorn apple", "077": "morning glory", "078": "passion flower",
    "079": "lotus", "080": "toad lily", "081": "bird of paradise", "082": "anthurium",
    "083": "frangipani", "084": "clematis", "085": "hibiscus", "086": "columbine",
    "087": "desert-rose", "088": "tree mallow", "089": "magnolia", "090": "cyclamen",
    "091": "watercress", "092": "monkshood", "093": "canna lily", "094": "hippeastrum",
    "095": "bee balm", "096": "ball moss", "097": "foxglove", "098": "bougainvillea",
    "099": "camellia", "100": "mallow", "101": "mexican petunia", "102": "bromelia"
}

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- HELPER FUNCTIONS ---

# NEW: Firebase Admin SDK Initialization
def initialize_firebase_admin_sdk():
    logging.info("Initializing Firebase Admin SDK...")
    try:
        # Load credentials from the environment variable (GitHub Secret)
        service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY') # <--- GET THE BASE64 STRING
        if service_account_base64 is None:
            logging.critical("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found. Cannot initialize Firebase.")
            raise ValueError("Firebase Service Account Key missing. Ensure it's set as a GitHub Secret.")

        # Decode the Base64 string back to JSON
        # <--- THIS IS THE NEW LINE FOR DECODING
        service_account_json_decoded = base64.b64decode(service_account_base64).decode('utf-8')
        
        # Load the JSON string into a Python dictionary
        # <--- USE THE DECODED STRING HERE
        cred = credentials.Certificate(json.loads(service_account_json_decoded))
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': FIREBASE_STORAGE_BUCKET
        })
        logging.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Firebase Admin SDK: {e}")
        raise

def download_user_data_from_firebase():
    """Downloads user-uploaded training data from Firebase Storage."""
    logging.info(f"📦 Checking for user-uploaded training data in Firebase Storage bucket '{FIREBASE_STORAGE_BUCKET}' under prefix '{FIREBASE_USER_DATA_PREFIX}'...")
    
    # Ensure the local directory exists
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    
    bucket = storage.bucket() # Get the storage bucket object

    try:
        # List all blobs (files) under the specified prefix
        blobs = bucket.list_blobs(prefix=FIREBASE_USER_DATA_PREFIX)
        downloaded_files_count = 0
        extracted_classes = set()

        for blob in blobs:
            # Skip "directory" blobs themselves (e.g., "user_training_data/class_name/")
            if blob.name.endswith('/'):
                continue
            
            # Construct local file path (e.g., user_training_data/rose/image1.jpg)
            relative_path = blob.name[len(FIREBASE_USER_DATA_PREFIX):]
            local_file_path = os.path.join(USER_DATA_DIR, relative_path)
            
            # Ensure the local directory structure exists for the file
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            try:
                blob.download_to_filename(local_file_path)
                downloaded_files_count += 1
                
                # Extract class name from the relative path (e.g., "rose" from "rose/image1.jpg")
                class_name = relative_path.split(os.sep)[0] # os.sep handles / or \
                if class_name:
                    extracted_classes.add(class_name)

                logging.debug(f"Downloaded: {blob.name} to {local_file_path}")
            except Exception as e:
                logging.warning(f"❌ Failed to download {blob.name}: {e}")
                
        if downloaded_files_count > 0:
            logging.info(f"✅ Downloaded {downloaded_files_count} user images for {len(extracted_classes)} classes from Firebase Storage.")
        else:
            logging.info("ℹ️ No user-uploaded data found in Firebase Storage under the specified prefix.")

    except Exception as e:
        logging.warning(f"❌ An error occurred during Firebase Storage download: {e}")


class UnifiedDataset(torch.utils.data.Dataset):
    """
    A custom dataset class to combine samples from different ImageFolder datasets
    and map them to a unified set of class indices.
    """
    def __init__(self, samples_with_original_class_strings, unified_class_string_to_idx, transform=None):
        self.samples = []
        for path, original_class_string in samples_with_original_class_strings:
            unified_idx = unified_class_string_to_idx.get(original_class_string)
            if unified_idx is not None:
                self.samples.append((path, unified_idx))
            else:
                logging.warning(f"Class '{original_class_string}' for image '{path}' not found in unified mapping. Skipping image.")
        
        self.transform = transform
        # The classes for this dataset are the sorted list of unique common names
        self.classes = sorted(list(unified_class_string_to_idx.keys()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target_idx = self.samples[idx]
        image = PIL.Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target_idx

def get_merged_datasets():
    """
    Loads a base dataset (always present in repo) and merges with user-uploaded datasets.
    The final class mapping and the trained model will reflect all classes from
    both the base data and user data.
    """
    logging.info("📂 Loading base and user datasets...")

    all_unique_common_names = set()
    all_raw_samples = [] # To store (image_path, class_string) tuples from all sources

    # --- Load Base Dataset ---
    base_dataset = None
    if os.path.exists(BASE_TRAINING_DATA_DIR) and os.listdir(BASE_TRAINING_DATA_DIR):
        try:
            # Use a temporary transform that only reads the image without normalization
            # The actual transform will be applied later in UnifiedDataset
            temp_transform = transforms.Compose([
                transforms.Resize((224, 224)), # Resize for consistency
                transforms.ToTensor() # Just to load it, not for normalization yet
            ])
            
            base_dataset = datasets.ImageFolder(BASE_TRAINING_DATA_DIR, transform=temp_transform)
            if base_dataset and len(base_dataset) > 0:
                logging.info(f"Loaded {len(base_dataset)} images from base data with {len(base_dataset.classes)} classes.")
                for path, original_idx in base_dataset.samples:
                    class_string = base_dataset.classes[original_idx]
                    all_raw_samples.append((path, class_string))
                    all_unique_common_names.add(class_string)
            else:
                logging.warning("Base dataset directory found, but it appears empty or could not load any images.")
                base_dataset = None
        except Exception as e:
            logging.warning(f"Could not load base dataset from '{BASE_TRAINING_DATA_DIR}': {e}. Ensure folder structure is correct (base_training_data/class_name/image.jpg).")
            base_dataset = None
    else:
        logging.warning(f"Base dataset directory '{BASE_TRAINING_DATA_DIR}' not found or is empty. Proceeding without it, but this is the initial data source.")

    # --- Load User Dataset ---
    user_dataset = None
    if os.path.exists(USER_DATA_DIR) and os.listdir(USER_DATA_DIR):
        try:
            temp_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            user_dataset = datasets.ImageFolder(USER_DATA_DIR, transform=temp_transform)
            if user_dataset and len(user_dataset) > 0:
                logging.info(f"Loaded {len(user_dataset)} images from user-uploaded data with {len(user_dataset.classes)} classes.")
                for path, original_idx in user_dataset.samples:
                    class_string = user_dataset.classes[original_idx]
                    all_raw_samples.append((path, class_string))
                    all_unique_common_names.add(class_string)
            else:
                logging.info("User dataset directory found, but it appears empty or could not load any images.")
                user_dataset = None
        except Exception as e:
            logging.warning(f"Could not load user dataset from '{USER_DATA_DIR}': {e}. Ensure folder structure is correct (user_training_data/class_name/image.jpg).")
            user_dataset = None
    else:
        logging.info(f"User dataset directory '{USER_DATA_DIR}' not found or is empty. Proceeding without it.")

    # --- Finalize Combined Dataset ---
    if not all_raw_samples: # Check if there's any data at all
        logging.critical("No base or user-uploaded classes/images found for training. Aborting.")
        raise ValueError("No data available for training. Ensure 'base_training_data' has images and/or user data is uploaded.")

    logging.info(f"✅ Total unique common names for training: {len(all_unique_common_names)}")

    # Sort the unique common names alphabetically to ensure consistent indexing
    final_sorted_common_names = sorted(list(all_unique_common_names))

    # Create a mapping from common name strings to new unified indices (0, 1, 2...)
    class_string_to_unified_idx = {name: i for i, name in enumerate(final_sorted_common_names)}

    # Create the unified dataset with the correct transforms
    full_unified_dataset = UnifiedDataset(
        samples_with_original_class_strings=all_raw_samples,
        unified_class_string_to_idx=class_string_to_unified_idx,
        transform=train_transform # Apply the full training transform here
    )

    # Validation split from the combined dataset
    train_size = int((1 - VALIDATION_SPLIT_RATIO) * len(full_unified_dataset))
    val_size = len(full_unified_dataset) - train_size
    
    # Handle very small datasets for train/val split
    if len(full_unified_dataset) < 2: # Need at least 2 images for a meaningful split
        logging.warning("Combined dataset has fewer than 2 images. Will use all for training, validation set will be empty.")
        train_size = len(full_unified_dataset)
        val_size = 0
    elif train_size == 0: # Ensure train_size is at least 1 if full_unified_dataset > 0
        train_size = 1
        val_size = len(full_unified_dataset) - train_size
        logging.warning("Training set would be 0, adjusted to 1 image. Validation set will be smaller.")
    
    train_subset, val_subset = random_split(full_unified_dataset, [train_size, val_size])

    final_train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    # Only create val_loader if val_subset is not empty
    final_val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1) if len(val_subset) > 0 else []

    logging.info(f"Training loader size: {len(train_subset)} images, Validation loader size: {len(val_subset)} images.")
    return final_train_loader, final_val_loader, final_sorted_common_names, len(final_sorted_common_names)


# --- Model Building, Training, and Saving ---
def build_model(num_classes):
    """Builds and initializes the ResNet model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(DEVICE)
    logging.info(f"Model built with {num_classes} output classes.")
    return model

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    """Trains the model."""
    logging.info("📚 Starting model training...")
    best_loss = float('inf') # Initialize best_loss to infinity
    best_model_path = "best_model_weights.pth" # Path to save the best model weights locally

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                logging.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            # Check if val_loader has data before iterating
            if val_loader and len(val_loader) > 0:
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            else:
                logging.info("Validation loader is empty. Skipping validation step for this epoch.")

        if val_loader and total_val > 0: # Only calculate and log validation metrics if there was data
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            logging.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path) # Save to the defined path
                logging.info(f"Saving best model weights with validation loss: {best_loss:.4f}")
        else:
            logging.info("No meaningful validation performed. Saving model weights after epoch.")
            # If no validation data, always save the model state after each epoch
            torch.save(model.state_dict(), best_model_path)


    logging.info("✅ Training complete.")
    # Load the best weights if they were saved
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded best model weights for final saving.")
    else:
        logging.warning("No 'best_model_weights.pth' found. Using the last epoch's model.")
    return model

# NEW: Function to upload files to Firebase Storage
def upload_to_firebase_storage(local_file_path, remote_blob_path):
    """Uploads a file to Firebase Storage."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(remote_blob_path)
        blob.upload_from_filename(local_file_path)
        logging.info(f"✅ Uploaded {local_file_path} to gs://{FIREBASE_STORAGE_BUCKET}/{remote_blob_path}")
    except Exception as e:
        logging.error(f"❌ Failed to upload {local_file_path} to Firebase Storage: {e}")
        raise

def save_model_and_mapping(model, classes):
    """Saves the TorchScript model and class-to-label mapping locally, then uploads to Firebase."""
    model.eval()

    # Ensure the model is on CPU for scripting if it was on GPU during training,
    # as scripted models are typically deployed to CPU environments.
    model_on_cpu = model.to('cpu')
    scripted_model = torch.jit.script(model_on_cpu)
    
    # Save model locally first
    scripted_model.save(MODEL_OUTPUT)
    logging.info(f"✅ Saved TorchScript model locally as {MODEL_OUTPUT}")

    # Save class mapping locally first
    idx_to_label = {i: label for i, label in enumerate(classes)}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(idx_to_label, f, indent=4)
    logging.info(f"✅ Saved class-to-label mapping locally to {CLASS_MAPPING_FILE}")

    # --- UPLOAD TO FIREBASE STORAGE ---
    logging.info("⬆️ Uploading model and mapping files to Firebase Storage...")
    upload_to_firebase_storage(MODEL_OUTPUT, os.path.join(FIREBASE_MODEL_UPLOAD_PREFIX, MODEL_OUTPUT))
    upload_to_firebase_storage(CLASS_MAPPING_FILE, os.path.join(FIREBASE_MODEL_UPLOAD_PREFIX, CLASS_MAPPING_FILE))

    # Increment model version (you'd need a model_version.txt in your repo or storage)
    # This part requires managing model_version.txt's current state.
    # A simple approach is to read, increment, then upload.
    # For initial setup, you might just create it.
    model_version_file = "model_version.txt"
    current_version = 0
    
    # NEW: Try to download model_version.txt from Firebase Storage first
    try:
        bucket = storage.bucket()
        blob = bucket.blob(os.path.join(FIREBASE_MODEL_UPLOAD_PREFIX, model_version_file))
        version_data = blob.download_as_text()
        current_version = int(version_data.strip())
        logging.info(f"Downloaded existing model_version.txt from Firebase: v{current_version}")
    except Exception as e:
        logging.warning(f"Could not download model_version.txt from Firebase or parse it: {e}. Starting version at 0 (or continuing from local if it exists).")
        # Fallback to local file if Firebase download failed or was not found
        if os.path.exists(model_version_file):
            try:
                with open(model_version_file, "r") as f:
                    current_version = int(f.read().strip())
                logging.info(f"Read local model_version.txt: v{current_version}")
            except ValueError:
                logging.warning(f"Could not read integer from local {model_version_file}. Starting version at 0.")
                current_version = 0
        else:
            current_version = 0 # Default to 0 if neither local nor remote found

    new_version = current_version + 1
    with open(model_version_file, "w") as f:
        f.write(str(new_version))
    logging.info(f"Incremented model version to {new_version}")

    upload_to_firebase_storage(model_version_file, os.path.join(FIREBASE_MODEL_UPLOAD_PREFIX, model_version_file))
    logging.info("✅ Model version uploaded to Firebase Storage.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    logging.info("DEBUG: Entering main execution block of auto_retrain.py.")

    # NEW: Initialize Firebase Admin SDK first
    try:
        initialize_firebase_admin_sdk()
    except Exception:
        logging.critical("Firebase Admin SDK initialization failed. Aborting script.")
        exit(1)

    # 1. Skip Oxford 102 Flower dataset download and preparation
    logging.info("DEBUG: Skipping Oxford data download and preparation as requested.")

    # 2. Download uploaded user data (NEW FUNCTION CALL)
    logging.info("DEBUG: Starting user data download from Firebase Storage.")
    download_user_data_from_firebase() # CALL THE NEW FUNCTION
    logging.info("DEBUG: User data download complete.")

    # 3. Load base and user datasets
    logging.info("DEBUG: Getting merged datasets (base + user).")
    try:
        train_loader, val_loader, all_classes, num_classes = get_merged_datasets()
        logging.info(f"DEBUG: Merged datasets loaded. Number of unique classes: {num_classes}")
        
        if num_classes == 0:
            logging.critical("No classes found in base or user data. Cannot train a model.")
            exit(1)
        if len(train_loader.dataset) == 0:
            logging.critical("Training dataset (base + user data) is empty. Cannot train a model.")
            exit(1)
    except ValueError as ve: # Catch the specific error from get_merged_datasets
        logging.critical(f"Failed to load datasets: {ve}. Aborting.")
        exit(1)
    except Exception as e: # Catch any other unexpected errors
        logging.critical(f"An unexpected error occurred while preparing datasets: {e}. Aborting.")
        exit(1)


    # 4. Build & train model
    logging.info("DEBUG: Building model.")
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info("DEBUG: Starting model training on base + user data.")
    trained_model = train_model(model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer)
    logging.info("DEBUG: Model training finished.")

    # 5. Save TorchScript model and class mapping locally, then upload to Firebase Storage
    logging.info("DEBUG: Saving model and mapping files locally and uploading to Firebase.")
    try:
        save_model_and_mapping(trained_model, all_classes)
        logging.info("DEBUG: Model and mapping files saved and uploaded.")
    except Exception as e:
        logging.critical(f"Failed to save or upload model/class mapping: {e}. Aborting.")
        exit(1)

    # Clean up downloaded files
    logging.info("DEBUG: Starting cleanup of raw data.")
    for f in ["102flowers.tgz", "imagelabels.mat", "setid.mat"]: # These are old files, likely not created now
        if os.path.exists(f):
            os.remove(f)
            logging.info(f"Cleaned up: {f}")
    if os.path.exists("jpg"): # This is also an old directory, likely not created now
        shutil.rmtree("jpg")
        logging.info("Cleaned up: jpg directory.")
    
    # Clean up user_training_data directory downloaded from Firebase
    if os.path.exists(USER_DATA_DIR):
        shutil.rmtree(USER_DATA_DIR)
        logging.info(f"Cleaned up: {USER_DATA_DIR} directory.")
    
    # Also clean up locally saved model and mapping files after upload if desired,
    # or keep them for debugging if the action runner automatically cleans up anyway.
    if os.path.exists(MODEL_OUTPUT):
        os.remove(MODEL_OUTPUT)
        logging.info(f"Cleaned up local {MODEL_OUTPUT}.")
    if os.path.exists(CLASS_MAPPING_FILE):
        os.remove(CLASS_MAPPING_FILE)
        logging.info(f"Cleaned up local {CLASS_MAPPING_FILE}.")
    if os.path.exists("best_model_weights.pth"): # Temp file created during training
        os.remove("best_model_weights.pth")
        logging.info("Cleaned up local best_model_weights.pth.")
    if os.path.exists("model_version.txt"): # This file is also uploaded
        os.remove("model_version.txt")
        logging.info("Cleaned up local model_version.txt.")

    logging.info("Cleanup complete. Script finished successfully.")
