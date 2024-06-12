import cv2
from camera import Camera
from recognizer import Recognizer
from threading import Thread

known_face_encodings = []
known_face_names = []

def main():
    camera = Camera()
    recognizer = Recognizer()
    input_thread = Thread(target=recognizer.handle_new_face, daemon=True)
    input_thread.start()
    
    while True:
        frame = camera.get_frame()
        
        if frame is None:
            break
        
        frame = recognizer.recognize_faces(frame)
        cv2.imshow('Webcam feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
