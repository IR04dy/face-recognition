import cv2
import numpy as np
from deepface import DeepFace
import time
import os

# Path to stored embeddings
embeddings_path = "embeddings"

# Load stored embeddings into memory
def load_reference_embeddings(embeddings_path):
    reference_embeddings = []
    for person_name in os.listdir(embeddings_path):
        person_folder = os.path.join(embeddings_path, person_name)
        if os.path.isdir(person_folder):
            for embedding_file in os.listdir(person_folder):
                embedding_path = os.path.join(person_folder, embedding_file)
                try:
                    # Load embedding and associate it with the person
                    embedding = np.load(embedding_path)
                    reference_embeddings.append({"name": person_name, "embedding": embedding})
                except Exception as e:
                    print(f"Error loading embedding {embedding_path}: {e}")
    return reference_embeddings

# Load all reference embeddings at the start
reference_embeddings = load_reference_embeddings(embeddings_path)

# Function to recognize faces in a frame
def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (160, 160))

        # Compare with reference embeddings
        try:
            embedding = DeepFace.represent(face_roi, model_name="Facenet")[0]["embedding"]
            best_match = None
            min_distance = float("inf")
            certainty = 0
            for ref in reference_embeddings:
                dist = np.linalg.norm(np.array(embedding) - np.array(ref["embedding"]))
                if dist < 8 and dist < min_distance:  # Threshold for recognition
                    min_distance = dist
                    best_match = ref["name"]
                    certainty = max(0, (8 - dist) / 8 * 100)

            label = f"{best_match} ({certainty:.1f}%)" if best_match else "Unknown"
        except Exception as e:
            label = "Error"

        if label != "Error":
            # Draw bounding box and label
            color = (0, int(certainty * 2.55), 255 - int(certainty * 2.55))  # Green for high, red for low
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    # Perform face recognition on the frame
    output_frame = recognize_faces(frame)
    # Display the resulting frame
    cv2.imshow('Live Face Recognition', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()