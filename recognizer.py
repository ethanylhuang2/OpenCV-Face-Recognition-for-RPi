import cv2
import numpy as np
import face_recognition
from threading import Thread, Lock
from queue import Queue
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='recognizer_log.txt', filemode='w')
logger = logging.getLogger()

class Recognizer:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(r'models/deploy.prototxt', r'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.known_face_encodings = []
        self.known_face_names = []
        self.unknown_face_encodings = []
        self.queue_lock = Lock()

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
            logger.debug(f"Known faces count: {len(self.known_face_encodings)}")
            logger.debug(f"Unknown faces count: {len(self.unknown_face_encodings)}")

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            else: 
                unknown_matches = face_recognition.compare_faces(self.unknown_face_encodings, encoding)
                if not self.unknown_face_encodings or False in unknown_matches:
                    logger.debug("Added to unknown face encodings")
                    self.unknown_face_encodings.append(encoding)
                else:
                    logger.debug("Face already in queue")

                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame
    
    def handle_new_face(self):
        while True:
            if len(self.unknown_face_encodings) > 0:
                name = input("Enter name: ").strip()
                if name:
                    with self.queue_lock:
                        encoding = self.unknown_face_encodings.pop(0)
                        self.known_face_names.append(name)
                        self.known_face_encodings.append(encoding)
                        logger.debug(f"Added {name} to known faces")
                        logger.debug(f"Known faces: {self.known_face_names}")

            time.sleep(0.1)