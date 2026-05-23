import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

# Load FaceNet model and MTCNN for face detection
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

# Load embeddings and names from .npz file
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'face_embeddings.npz')
if os.path.exists(EMBEDDINGS_PATH):
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    known_embeddings = data['embeddings']
    known_names = data['names']
else:
    known_embeddings = np.array([])
    known_names = []

def recognize_face(face_img):
    aligned = mtcnn(Image.fromarray(face_img))
    if aligned is None:
        return None, None
    if aligned.ndim == 3:
        aligned = aligned.unsqueeze(0)
    emb = facenet(aligned).detach().numpy()[0]

    if len(known_embeddings) == 0:
        return None, None
    dists = np.linalg.norm(known_embeddings - emb, axis=1)
    min_idx = np.argmin(dists)


    if dists[min_idx] < 1.1:  
        return known_names[min_idx], dists[min_idx]
    else:
        return None, dists[min_idx]
