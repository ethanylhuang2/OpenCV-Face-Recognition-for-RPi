import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)  # 0 is typically the default camera
        
        if not self.cap.isOpened():
            print("Error: Could not open video capture device.")
            return

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()