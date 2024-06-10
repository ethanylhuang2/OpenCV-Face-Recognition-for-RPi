import cv2
import numpy as np
import face_recognition
from threading import Thread, Lock

class Recognizer:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(r'models/deploy.prototxt', r'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.known_face_encodings = []
        self.known_face_names = []
        self.lock = Lock()

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        locations = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append(frame[startY:endY, startX:endX])
                locations.append((startY, endX, endY, startX))
                    
        return faces, locations
    
    def recognize_faces(self, frame):
        faces, locations = self.detect_faces(frame)
        face_encodings = face_recognition.face_encodings(frame, locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            else:
                input_thread = Thread(target=self.handle_new_face, args=(encoding,), daemon=True)
                input_thread.start()
                #self.known_face_encodings.append(encoding)
                #name = f"Person {self.next_person_id}"
                #self.known_face_names.append(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame
    
    def handle_new_face(self,encoding):
        name = input("Enter name: ").strip()
        if name:
            with self.lock:
                self.known_face_names.append(name)
                self.known_face_encodings.append(encoding)
                print(f"Added {name} to known faces")