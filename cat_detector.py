"""
Cat Detection using Pre-trained YOLOv8
"""

from ultralytics import YOLO
import cv2
import os

class CatDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the cat detector with pre-trained YOLOv8 model
        """
        print(f"Loading pre-trained {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
    
    def detect_cats_in_image(self, image_path, confidence=0.5):
        """
        Detect cats in a single image
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        results = self.model.predict(image_path, conf=confidence)
        return results
    
    def detect_cats_in_video(self, video_path, confidence=0.5):
        """
        Detect cats in a video file
        """
        results = self.model.predict(video_path, conf=confidence)
        return results
    
    def detect_cats_live(self, confidence=0.5):
        """
        Real-time cat detection from webcam
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            annotated_frame = results[0].plot()
            
            cv2.imshow('Cat Detection - Press Q to quit', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    detector = CatDetector('yolov8n.pt')
    
    # For image detection
    # results = detector.detect_cats_in_image('cat_image.jpg')
    
    # For video detection
    # results = detector.detect_cats_in_video('cat_video.mp4')
    
    # For live camera detection
    # detector.detect_cats_live()
    
    print("Cat detector ready! Use the methods above for detection.")
