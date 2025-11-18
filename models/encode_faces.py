import os
import numpy as np
from PIL import Image
import torch
import logging

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project's base directory (yolo/)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

FACES_DIR = os.path.join(BASE_DIR, 'faces')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'face_embeddings.npz')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Initialization ---
# Moved inside the main function to avoid loading models if not run as script
from facenet_pytorch import InceptionResnetV1, MTCNN

# --- Main Encoding Logic ---
def encode_faces():
    """
    Scans face directories, generates a single averaged embedding for each person
    (folder), and saves them to a .npz file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(keep_all=True, device=device) # keep_all=True to process multiple images

    final_embeddings = []
    final_names = []

    if not os.path.isdir(FACES_DIR):
        logging.error(f"The main 'faces' directory was not found at: {FACES_DIR}")
        return

    logging.info(f"Scanning for faces in: {FACES_DIR}")

    for person_name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            person_embeddings = []
            logging.info(f"--- Processing directory for '{person_name}' ---")
            
            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                logging.warning(f"  - No images found for '{person_name}', skipping.")
                continue

            for fname in image_files:
                fpath = os.path.join(person_dir, fname)
                try:
                    img = Image.open(fpath).convert('RGB')
                    # Detect all faces in the image
                    face_tensors = mtcnn(img)
                    
                    if face_tensors is not None:
                        with torch.no_grad():
                            # Get embeddings for all detected faces
                            embeddings = facenet(face_tensors.to(device))
                            person_embeddings.extend(embeddings.cpu())
                        logging.info(f"  - Extracted {len(face_tensors)} face(s) from {fname}")
                    else:
                        logging.warning(f"  - No face detected in {fname}, skipping.")
                except Exception as e:
                    logging.error(f"  - Error processing {fname}: {e}")

            if person_embeddings:
                # Average all collected embeddings for this person into one
                avg_embedding = torch.stack(person_embeddings).mean(dim=0)
                final_embeddings.append(avg_embedding.numpy())
                final_names.append(person_name)
                logging.info(f"--- Finished '{person_name}'. Averaged {len(person_embeddings)} embeddings. ---")

    # Save the final averaged embeddings
    if final_embeddings:
        np.savez(EMBEDDINGS_PATH, embeddings=np.array(final_embeddings), names=np.array(final_names))
        logging.info(f"Successfully saved {len(final_names)} averaged face embeddings to {EMBEDDINGS_PATH}")
    else:
        logging.warning("No faces were encoded. The output file was not created.")

if __name__ == "__main__":
    encode_faces()
