#!/usr/bin/env python3
"""
Face utility functions for facial landmark detection.

This module contains helper functions for working with dlib face detection
and landmark prediction results, converting between different coordinate formats.
"""

import numpy as np
import cv2


class FaceUtils:
    """
    Utility class for face detection and landmark processing operations.
    """
    
    @staticmethod
    def rect_to_bb(rect):
        """
        Convert a dlib rectangle to a bounding box tuple.
        
        Args:
            rect: dlib rectangle object
            
        Returns:
            tuple: (x, y, w, h) bounding box coordinates
        """
        # Extract the starting and ending (x, y)-coordinates of the
        # bounding box
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        # Return a tuple of (x, y, w, h)
        return (x, y, w, h)
    
    @staticmethod
    def shape_to_np(shape, dtype="int"):
        """
        Convert a dlib shape object to a NumPy array.
        
        Args:
            shape: dlib shape object containing facial landmarks
            dtype: data type for the output array
            
        Returns:
            numpy.ndarray: Array of (x, y) coordinates with shape (68, 2)
        """
        # Initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        
        # Loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        
        # Return the list of (x, y)-coordinates
        return coords
    
    @staticmethod
    def bb_to_rect(bb):
        """
        Convert a bounding box tuple to dlib rectangle format.
        
        Args:
            bb: tuple of (x, y, w, h) bounding box coordinates
            
        Returns:
            dlib.rectangle: dlib rectangle object
        """
        import dlib
        x, y, w, h = bb
        return dlib.rectangle(x, y, x + w, y + h)
    
    @staticmethod
    def resize_landmarks(landmarks, original_size, new_size):
        """
        Resize landmark coordinates when image is resized.
        
        Args:
            landmarks (numpy.ndarray): Original landmark coordinates
            original_size (tuple): Original image size (width, height)
            new_size (tuple): New image size (width, height)
            
        Returns:
            numpy.ndarray: Resized landmark coordinates
        """
        orig_w, orig_h = original_size
        new_w, new_h = new_size
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        resized_landmarks = landmarks.copy()
        resized_landmarks[:, 0] *= scale_x
        resized_landmarks[:, 1] *= scale_y
        
        return resized_landmarks.astype(int)
    
    @staticmethod
    def extract_face_region(image, landmarks, region_name, padding=10):
        """
        Extract a specific facial region based on landmarks.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Facial landmark coordinates
            region_name (str): Name of the region to extract
            padding (int): Padding around the region
            
        Returns:
            numpy.ndarray: Cropped image of the facial region
        """
        # Define landmark indices for each facial region
        regions = {
            'left_eye': list(range(42, 48)),
            'right_eye': list(range(36, 42)),
            'nose': list(range(27, 36)),
            'mouth': list(range(48, 68)),
            'left_eyebrow': list(range(22, 27)),
            'right_eyebrow': list(range(17, 22)),
            'jaw': list(range(0, 17)),
            'full_face': list(range(0, 68))
        }
        
        if region_name not in regions:
            raise ValueError(f"Unknown region: {region_name}")
        
        # Get landmarks for the specified region
        region_landmarks = landmarks[regions[region_name]]
        
        # Calculate bounding box
        x_min = np.min(region_landmarks[:, 0]) - padding
        y_min = np.min(region_landmarks[:, 1]) - padding
        x_max = np.max(region_landmarks[:, 0]) + padding
        y_max = np.max(region_landmarks[:, 1]) + padding
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Extract region
        region_image = image[y_min:y_max, x_min:x_max]
        
        return region_image, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    @staticmethod
    def calculate_face_metrics(landmarks):
        """
        Calculate various face metrics from landmarks.
        
        Args:
            landmarks (numpy.ndarray): Facial landmark coordinates
            
        Returns:
            dict: Dictionary containing various face measurements
        """
        # Eye centers
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        
        # Inter-ocular distance
        inter_ocular_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Face width (jaw points)
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        
        # Face height (top of forehead to bottom of chin)
        # Approximate forehead point
        nose_bridge_top = landmarks[27]
        forehead_approx = nose_bridge_top + [0, -inter_ocular_distance * 0.5]
        chin_bottom = landmarks[8]  # Bottom of chin
        face_height = np.linalg.norm(forehead_approx - chin_bottom)
        
        # Nose dimensions
        nose_width = np.linalg.norm(landmarks[31] - landmarks[35])
        nose_height = np.linalg.norm(landmarks[27] - landmarks[33])
        
        # Mouth dimensions
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        
        return {
            'inter_ocular_distance': float(inter_ocular_distance),
            'face_width': float(face_width),
            'face_height': float(face_height),
            'nose_width': float(nose_width),
            'nose_height': float(nose_height),
            'mouth_width': float(mouth_width),
            'mouth_height': float(mouth_height),
            'left_eye_center': left_eye_center.tolist(),
            'right_eye_center': right_eye_center.tolist()
        }
    
    @staticmethod
    def detect_facial_features(landmarks):
        """
        Analyze facial features and characteristics from landmarks.
        
        Args:
            landmarks (numpy.ndarray): Facial landmark coordinates
            
        Returns:
            dict: Analysis results including feature characteristics
        """
        metrics = FaceUtils.calculate_face_metrics(landmarks)
        
        # Analyze face shape based on ratios
        face_ratio = metrics['face_height'] / metrics['face_width']
        
        if face_ratio > 1.3:
            face_shape = 'oval/long'
        elif face_ratio < 1.1:
            face_shape = 'round/square'
        else:
            face_shape = 'balanced'
        
        # Eye analysis
        left_eye_landmarks = landmarks[42:48]
        right_eye_landmarks = landmarks[36:42]
        
        left_eye_width = np.linalg.norm(left_eye_landmarks[0] - left_eye_landmarks[3])
        left_eye_height = np.linalg.norm(left_eye_landmarks[1] - left_eye_landmarks[5])
        
        right_eye_width = np.linalg.norm(right_eye_landmarks[0] - right_eye_landmarks[3])
        right_eye_height = np.linalg.norm(right_eye_landmarks[1] - right_eye_landmarks[5])
        
        avg_eye_ratio = ((left_eye_width / left_eye_height) + (right_eye_width / right_eye_height)) / 2
        
        if avg_eye_ratio > 3.0:
            eye_shape = 'narrow/almond'
        elif avg_eye_ratio < 2.5:
            eye_shape = 'round/wide'
        else:
            eye_shape = 'normal'
        
        return {
            'face_shape': face_shape,
            'face_ratio': face_ratio,
            'eye_shape': eye_shape,
            'eye_ratio': avg_eye_ratio,
            'measurements': metrics
        }
    
    @staticmethod
    def landmarks_to_mask(landmarks, image_shape, region_name=None):
        """
        Create a binary mask from facial landmarks.
        
        Args:
            landmarks (numpy.ndarray): Facial landmark coordinates
            image_shape (tuple): Shape of the image (height, width)
            region_name (str, optional): Specific region to create mask for
            
        Returns:
            numpy.ndarray: Binary mask image
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        if region_name is None:
            # Create mask for entire face using convex hull
            hull = cv2.convexHull(landmarks)
            cv2.fillPoly(mask, [hull], 255)
        else:
            # Create mask for specific region
            regions = {
                'left_eye': landmarks[42:48],
                'right_eye': landmarks[36:42],
                'nose': landmarks[27:36],
                'mouth': landmarks[48:68],
                'left_eyebrow': landmarks[22:27],
                'right_eyebrow': landmarks[17:22],
                'jaw': landmarks[0:17]
            }
            
            if region_name in regions:
                hull = cv2.convexHull(regions[region_name])
                cv2.fillPoly(mask, [hull], 255)
        
        return mask