#!/usr/bin/env python3
"""
Facial Landmark Detection using dlib and OpenCV

This script detects 68 facial landmarks in images including:
- Eyes (left and right)
- Eyebrows (left and right) 
- Nose
- Mouth
- Jawline

Based on the PyImageSearch tutorial and dlib's facial landmark detector.
"""

import argparse
import cv2
import dlib
import numpy as np
import os
import sys
import json
from pathlib import Path
import yaml

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualization'))

from src.utils.face_utils import FaceUtils
from src.visualization.visualizer import LandmarkVisualizer


class FacialLandmarkDetector:
    """
    A class for detecting and processing facial landmarks in images.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the facial landmark detector.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.face_detector = None
        self.landmark_predictor = None
        self.face_utils = FaceUtils()
        self.visualizer = LandmarkVisualizer()
        
        self._initialize_detectors()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        default_config = {
            'model': {
                'shape_predictor_path': 'models/shape_predictor_68_face_landmarks.dat',
                'face_detector': 'hog'
            },
            'processing': {
                'resize_width': 500,
                'upsampling_layers': 1
            },
            'visualization': {
                'face_box_color': [0, 255, 0],
                'landmark_color': [0, 0, 255],
                'landmark_radius': 2,
                'box_thickness': 2
            },
            'output': {
                'save_annotated': True,
                'output_format': 'jpg',
                'quality': 95
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                else:
                    config[key] = {**default_config[key], **config[key]}
        else:
            config = default_config
            
        return config
    
    def _initialize_detectors(self):
        """Initialize face detector and landmark predictor."""
        # Initialize face detector
        if self.config['model']['face_detector'] == 'hog':
            self.face_detector = dlib.get_frontal_face_detector()
        else:
            raise ValueError(f"Unsupported face detector: {self.config['model']['face_detector']}")
        
        # Initialize landmark predictor
        predictor_path = self.config['model']['shape_predictor_path']
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Landmark predictor model not found at: {predictor_path}")
        
        self.landmark_predictor = dlib.shape_predictor(predictor_path)
    
    def detect_faces(self, image):
        """
        Detect faces in the input image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of face rectangles
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, self.config['processing']['upsampling_layers'])
        return faces
    
    def detect_landmarks(self, image, face_rect):
        """
        Detect facial landmarks for a given face rectangle.
        
        Args:
            image (numpy.ndarray): Input image
            face_rect: dlib rectangle object representing face location
            
        Returns:
            numpy.ndarray: Array of (x, y) landmark coordinates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmark_predictor(gray, face_rect)
        landmarks_np = self.face_utils.shape_to_np(landmarks)
        return landmarks_np
    
    def process_image(self, image_path, output_path=None, save_landmarks=False):
        """
        Process an image to detect faces and facial landmarks.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save annotated image
            save_landmarks (bool): Whether to save landmarks as JSON
            
        Returns:
            dict: Processing results including faces, landmarks, and annotated image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        # Resize image
        original_height, original_width = image.shape[:2]
        if self.config['processing']['resize_width']:
            aspect_ratio = original_height / original_width
            new_width = self.config['processing']['resize_width']
            new_height = int(new_width * aspect_ratio)
            image = cv2.resize(image, (new_width, new_height))
        
        # Detect faces
        faces = self.detect_faces(image)
        
        results = {
            'image_path': image_path,
            'original_size': (original_width, original_height),
            'processed_size': image.shape[:2][::-1],
            'num_faces': len(faces),
            'faces': [],
            'annotated_image': image.copy()
        }
        
        # Process each face
        for i, face_rect in enumerate(faces):
            # Detect landmarks
            landmarks = self.detect_landmarks(image, face_rect)
            
            # Convert dlib rectangle to bounding box
            bbox = self.face_utils.rect_to_bb(face_rect)
            
            face_data = {
                'face_id': i + 1,
                'bbox': bbox,
                'landmarks': landmarks.tolist(),
                'landmark_regions': self.get_landmark_regions(landmarks)
            }
            
            results['faces'].append(face_data)
            
            # Annotate image
            results['annotated_image'] = self.visualizer.draw_face_box(
                results['annotated_image'], bbox, f"Face #{i + 1}"
            )
            results['annotated_image'] = self.visualizer.draw_landmarks(
                results['annotated_image'], landmarks
            )
            results['annotated_image'] = self.visualizer.draw_landmark_numbers(
                results['annotated_image'], landmarks
            )

            # Create white image and plot landmarks with numbers
            white_img = 255 * np.ones_like(results['annotated_image'])
            white_img = self.visualizer.draw_landmarks(white_img, landmarks)
            white_img = self.visualizer.draw_landmark_numbers(white_img, landmarks, color=(0,0,0))

            # Save white image
            white_img_path = os.path.splitext(image_path)[0] + '_landmarks_white.jpg'
            cv2.imwrite(white_img_path, white_img)
            print(f"White landmark image saved to: {white_img_path}")

            # Display white image
            cv2.imshow('Landmarks on White', white_img)
        
        # Save results
        if output_path and self.config['output']['save_annotated']:
            cv2.imwrite(output_path, results['annotated_image'])
            print(f"Annotated image saved to: {output_path}")
        
        if save_landmarks:
            landmarks_path = self._get_landmarks_path(image_path, output_path)
            with open(landmarks_path, 'w') as f:
                json.dump({
                    'image_path': image_path,
                    'faces': results['faces']
                }, f, indent=2)
            print(f"Landmarks saved to: {landmarks_path}")
        
        return results
    
    def get_landmark_regions(self, landmarks):
        """
        Extract specific facial regions from landmarks.
        
        Args:
            landmarks (numpy.ndarray): Array of landmark coordinates
            
        Returns:
            dict: Dictionary containing coordinates for each facial region
        """
        regions = {
            'jaw': landmarks[0:17].tolist(),
            'right_eyebrow': landmarks[17:22].tolist(),
            'left_eyebrow': landmarks[22:27].tolist(),
            'nose': landmarks[27:36].tolist(),
            'right_eye': landmarks[36:42].tolist(),
            'left_eye': landmarks[42:48].tolist(),
            'mouth': landmarks[48:68].tolist(),
        }
        return regions
    
    def _get_landmarks_path(self, image_path, output_path):
        """Generate path for saving landmarks JSON file."""
        if output_path:
            base_name = os.path.splitext(output_path)[0]
            return f"{base_name}_landmarks.json"
        else:
            base_name = os.path.splitext(image_path)[0]
            return f"{base_name}_landmarks.json"


def main():
    """Main function to run facial landmark detection from command line."""
    parser = argparse.ArgumentParser(description='Detect facial landmarks in images')
    parser.add_argument('-i', '--image', required=True,
                      help='Path to input image')
    parser.add_argument('-o', '--output', 
                      help='Path to output annotated image')
    parser.add_argument('-p', '--shape-predictor',
                      help='Path to facial landmark predictor model')
    parser.add_argument('-c', '--config',
                      help='Path to configuration file')
    parser.add_argument('--save-landmarks', action='store_true',
                      help='Save landmarks to JSON file')
    parser.add_argument('--no-display', action='store_true',
                      help='Do not display the result image')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = FacialLandmarkDetector(config_path=args.config)
        
        # Override model path if specified
        if args.shape_predictor:
            detector.config['model']['shape_predictor_path'] = args.shape_predictor
            detector._initialize_detectors()
        
        # Process image
        results = detector.process_image(
            args.image, 
            args.output, 
            save_landmarks=args.save_landmarks
        )
        
        # Print results
        print(f"\nFacial Landmark Detection Results:")
        print(f"Image: {args.image}")
        print(f"Faces detected: {results['num_faces']}")
        
        for face in results['faces']:
            print(f"\nFace #{face['face_id']}:")
            print(f"  Bounding box: {face['bbox']}")
            print(f"  Landmarks detected: {len(face['landmarks'])}")
            
            # Print landmark regions
            for region, points in face['landmark_regions'].items():
                print(f"  {region.replace('_', ' ').title()}: {len(points)} points")
        
        # Display result
        if not args.no_display:
            cv2.imshow('Facial Landmarks', results['annotated_image'])
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()