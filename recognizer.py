import cv2
import numpy as np
import face_recognition
from threading import Thread, Event

class Recognizer:
    def __init__(self, known_face_encodings, known_face_names):
        self.net = cv2.dnn.readNetFromCaffe(r'models/deploy.prototxt', r'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.new_face_locations = []
        self.new_face_encodings = []
        self.new_face_event = Event()

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
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append(frame[startY:endY, startX:endX])
                locations.append((startY, endX, endY, startX))
                    
        return faces, locations
    
    def recognize_faces(self, frame):
        faces, locations = self.detect_faces(frame)
        face_encodings = face_recognition.face_encodings(frame, locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if any(matches):
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            else:
                self.new_face_encodings.append(face_encoding)
                self.new_face_locations.append(locations[face_encodings.index(face_encoding)])
                self.new_face_event.set()

            face_names.append(name)

        for (top, right, bottom, left), name in zip(locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

    def new_face_input(self):
        while True:
            self.new_face_event.wait()
            self.new_face_event.clear()
            while self.new_face_encodings:
                face_encoding = self.new_face_encodings.pop(0)
                location = self.new_face_locations.pop(0)
                name = input("Enter name for the new face: ") or "Unknown"
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)