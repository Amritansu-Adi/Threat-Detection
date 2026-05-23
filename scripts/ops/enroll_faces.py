import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os
import time

# --- Configuration ---
SAVE_PATH = 'known_faces.pt'
CAPTURE_DELAY_SECONDS = 2
NUM_SAMPLES = 15
MIN_CONFIDENCE = 0.95

# --- Initialization ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=True
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print('Face detection and recognition models loaded.')

# --- Main Enrollment Logic ---
def enroll_face():
    """
    Captures face samples from the webcam, generates an average embedding,
    and saves it with a user-provided name.
    """
    # Clear existing file to start fresh
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"Removed existing '{SAVE_PATH}' for re-enrollment.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    name = input("Enter the name for the person being enrolled: ")
    if not name:
        print("Enrollment cancelled: Name cannot be empty.")
        cap.release()
        return

    print(f"\nStarting enrollment for '{name}'.")
    print(f"Please look at the camera. Capturing will begin in {CAPTURE_DELAY_SECONDS} seconds.")
    time.sleep(CAPTURE_DELAY_SECONDS)

    face_samples = []
    captured_count = 0
    while captured_count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect face
        boxes, conf = mtcnn.detect(frame)

        display_frame = frame.copy()

        if boxes is not None and conf is not None and len(conf) > 0 and conf[0] is not None and conf[0] > MIN_CONFIDENCE:
            # Draw bounding box for the first detected face
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display capture progress
            progress_text = f'Capturing sample {captured_count + 1}/{NUM_SAMPLES}'
            cv2.putText(display_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert frame for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Get cropped face tensor
            face_tensors = mtcnn(img)
            if face_tensors is not None:
                # In case mtcnn returns multiple faces, take the first one
                face_samples.append(face_tensors[0])
                captured_count += 1
                print(f"  - Sample {captured_count} captured.")
                # Pause briefly to allow for slight pose changes
                time.sleep(0.2)
        else:
            cv2.putText(display_frame, 'No face or low confidence', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Enrollment - Press "q" to cancel', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(face_samples) < NUM_SAMPLES:
        print("\nEnrollment failed: Could not capture enough face samples.")
        return

    print("\nAll samples collected. Generating embedding...")
    
    # --- Generate and Save Embedding ---
    try:
        # Stack samples into a batch for efficient processing
        samples_tensor = torch.stack(face_samples).to(device)
        
        with torch.no_grad():
            embeddings = resnet(samples_tensor)
        
        # Average the embeddings to get a single, robust representation and squeeze it
        avg_embedding = torch.mean(embeddings, dim=0).squeeze().cpu()
        
        # Create a new data structure
        known_faces_data = {"names": [], "embeddings": []}
            
        # Add new person's data
        known_faces_data["names"].append(name)
        known_faces_data["embeddings"].append(avg_embedding)
        
        # Save to disk
        torch.save(known_faces_data, SAVE_PATH)
        
        print(f"\nSuccess! Enrollment complete for '{name}'.")
        print(f"Data saved to '{SAVE_PATH}'.")

    except Exception as e:
        print(f"\nAn error occurred while generating the embedding: {e}")


if __name__ == "__main__":
    enroll_face()
