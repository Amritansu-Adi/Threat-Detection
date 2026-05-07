import torch
import numpy as np
import os
import logging
import cv2
from PIL import Image

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'face_embeddings.npz')
UNKNOWN_THRESHOLD = 0.9
HOMEOWNER_THRESHOLD = 1.00  # More lenient for homeowner
MIN_MARGIN_TO_SECOND_BEST = 0.04

# --- Global State for Face Data ---
known_face_embeddings = []
known_face_names = []

def load_known_faces():
    """
    Loads pre-computed, averaged face embeddings from the .npz file.
    """
    global known_face_embeddings, known_face_names
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
            # Embeddings are already averaged, just load them
            known_face_embeddings = [torch.from_numpy(emb) for emb in data['embeddings']]
            known_face_names = data['names'].tolist()
            logging.info(f"Loaded {len(known_face_names)} pre-computed, averaged face embeddings from {EMBEDDINGS_PATH}")
        except Exception as e:
            logging.error(f"Failed to load embeddings file at {EMBEDDINGS_PATH}: {e}")
    else:
        logging.warning(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        logging.warning("Please run models/encode_faces.py to generate embeddings.")

def recognize_face(face_img_bgr, mtcnn_model, resnet_model, device):
    """
    Recognize a face from a BGR image crop using pre-loaded, averaged embeddings.
    """
    if face_img_bgr is None or face_img_bgr.size == 0:
        return None, float('inf')

    if not known_face_embeddings:
        return None, float('inf')

    try:
        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(face_img_rgb)

        with torch.no_grad():
            # We only need the single best face from the live crop
            face_tensor = mtcnn_model(img_pil, save_path=None) # Use the single-face version of mtcnn
            if face_tensor is None:
                return None, float('inf')

            face_tensor = face_tensor.unsqueeze(0).to(device)
            embedding = resnet_model(face_tensor).squeeze().cpu()

            # Compare the live embedding against the pre-averaged known embeddings
            distances = [(embedding - known_emb).norm().item() for known_emb in known_face_embeddings]
            
            if not distances:
                return None, float('inf')

            sorted_indices = np.argsort(distances)
            min_idx = int(sorted_indices[0])
            min_dist = float(distances[min_idx])
            second_best = float(distances[int(sorted_indices[1])]) if len(sorted_indices) > 1 else float("inf")
            
            recognized_name = known_face_names[min_idx]
            threshold = HOMEOWNER_THRESHOLD if recognized_name.lower() == 'homeowner' else UNKNOWN_THRESHOLD

            margin_ok = (second_best - min_dist) >= MIN_MARGIN_TO_SECOND_BEST
            if min_dist < threshold and margin_ok:
                return recognized_name, min_dist
            else:
                return None, min_dist

    except Exception as e:
        # logging.error(f"Error during face recognition: {e}")
        return None, float('inf')

# Initial load of known faces when the module is imported
load_known_faces()
