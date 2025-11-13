import os
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

FACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'faces')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'face_embeddings.npz')

facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

known_embeddings = []
known_names = []

for root, dirs, files in os.walk(FACES_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(root, fname)
            label = os.path.basename(os.path.dirname(fpath))
            img = Image.open(fpath).convert('RGB')
            aligned = mtcnn(img)
            if aligned is not None:
                if aligned.ndim == 3:
                    aligned = aligned.unsqueeze(0)  # [1, 3, 160, 160]
                emb = facenet(aligned).detach().numpy()[0]
                known_embeddings.append(emb)
                known_names.append(label)
            else:
                print(f"No face detected in {fpath}, skipping.")

known_embeddings = np.array(known_embeddings)
np.savez(EMBEDDINGS_PATH, embeddings=known_embeddings, names=known_names)
print(f"Saved {len(known_names)} face embeddings to {EMBEDDINGS_PATH}")
