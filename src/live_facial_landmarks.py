import cv2
import sys
import os
from facial_landmarks import FacialLandmarkDetector

# Optional: allow config/model override via command line
import argparse
parser = argparse.ArgumentParser(description='Live facial landmark detection from webcam')
parser.add_argument('-c', '--config', help='Path to configuration file')
parser.add_argument('-p', '--shape-predictor', help='Path to facial landmark predictor model')
args = parser.parse_args()

# Initialize detector
facial_detector = FacialLandmarkDetector(config_path=args.config)
if args.shape_predictor:
    facial_detector.config['model']['shape_predictor_path'] = args.shape_predictor
    facial_detector._initialize_detectors()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open webcam.')
    sys.exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to capture frame from webcam.')
        break

    # Optionally resize frame for speed
    display_frame = frame.copy()
    faces = facial_detector.detect_faces(display_frame)

    for i, face_rect in enumerate(faces):
        landmarks = facial_detector.detect_landmarks(display_frame, face_rect)
        bbox = facial_detector.face_utils.rect_to_bb(face_rect)
        display_frame = facial_detector.visualizer.draw_face_box(display_frame, bbox, f"Face #{i+1}")
        display_frame = facial_detector.visualizer.draw_landmarks(display_frame, landmarks)
        display_frame = facial_detector.visualizer.draw_landmark_numbers(display_frame, landmarks)

    cv2.imshow('Live Facial Landmarks', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 