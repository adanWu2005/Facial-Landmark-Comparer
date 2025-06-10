#!/usr/bin/env python3
"""
Visualization utilities for facial landmark detection.

This module provides functions to visualize facial landmarks, face bounding boxes,
and other facial features on images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class LandmarkVisualizer:
    """
    Class for visualizing facial landmarks and face detection results.
    """
    
    def __init__(self):
        # Define colors for different facial regions
        self.region_colors = {
            'jaw': (255, 255, 0),        # Cyan
            'right_eyebrow': (0, 255, 255),  # Yellow
            'left_eyebrow': (0, 255, 255),   # Yellow
            'nose': (255, 0, 255),       # Magenta
            'right_eye': (0, 255, 0),    # Green
            'left_eye': (0, 255, 0),     # Green
            'mouth': (0, 0, 255),        # Red
        }
        
        # Define landmark point ranges for each region
        self.landmark_regions = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth': list(range(48, 68)),
        }
    
    def draw_landmarks(self, image, landmarks, color=(0, 0, 255), radius=2):
        """
        Draw facial landmarks on an image.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            color (tuple): BGR color for landmarks
            radius (int): Radius of landmark points
            
        Returns:
            numpy.ndarray: Image with landmarks drawn
        """
        image_copy = image.copy()
        
        # Draw each landmark point
        for (x, y) in landmarks:
            cv2.circle(image_copy, (int(x), int(y)), radius, color, -1)
        
        return image_copy
    
    def draw_landmarks_by_region(self, image, landmarks, show_numbers=False):
        """
        Draw facial landmarks with different colors for each facial region.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            show_numbers (bool): Whether to show landmark numbers
            
        Returns:
            numpy.ndarray: Image with colored landmarks drawn
        """
        image_copy = image.copy()
        
        # Draw landmarks for each region with different colors
        for region_name, indices in self.landmark_regions.items():
            color = self.region_colors[region_name]
            region_landmarks = landmarks[indices]
            
            for i, (x, y) in enumerate(region_landmarks):
                cv2.circle(image_copy, (int(x), int(y)), 2, color, -1)
                
                if show_numbers:
                    point_num = indices[i]
                    cv2.putText(image_copy, str(point_num), 
                              (int(x) + 3, int(y) - 3),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return image_copy
    
    def draw_face_box(self, image, bbox, label="Face", color=(0, 255, 0), thickness=2):
        """
        Draw a bounding box around a detected face.
        
        Args:
            image (numpy.ndarray): Input image
            bbox (tuple): Bounding box coordinates (x, y, w, h)
            label (str): Label text for the face
            color (tuple): BGR color for the bounding box
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with face bounding box drawn
        """
        image_copy = image.copy()
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label
        cv2.putText(image_copy, label, (x - 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return image_copy
    
    def draw_face_contours(self, image, landmarks, color=(255, 0, 0), thickness=1):
        """
        Draw contour lines connecting facial landmarks.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            color (tuple): BGR color for contour lines
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with face contours drawn
        """
        image_copy = image.copy()
        
        # Define contour connections for each facial region
        contour_connections = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_bottom': list(range(31, 36)),
            'right_eye': list(range(36, 42)) + [36],  # Close the loop
            'left_eye': list(range(42, 48)) + [42],   # Close the loop
            'outer_mouth': list(range(48, 60)) + [48], # Close the loop
            'inner_mouth': list(range(60, 68)) + [60]  # Close the loop
        }
        
        # Draw contour lines
        for region, indices in contour_connections.items():
            if len(indices) > 1:
                points = landmarks[indices]
                for i in range(len(points) - 1):
                    pt1 = tuple(map(int, points[i]))
                    pt2 = tuple(map(int, points[i + 1]))
                    cv2.line(image_copy, pt1, pt2, color, thickness)
        
        return image_copy
    
    def create_landmark_overlay(self, image, landmarks, alpha=0.7):
        """
        Create a semi-transparent overlay showing facial regions.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            alpha (float): Transparency factor (0-1)
            
        Returns:
            numpy.ndarray: Image with landmark overlay
        """
        overlay = image.copy()
        
        # Create masks for each facial region
        for region_name, indices in self.landmark_regions.items():
            if len(indices) < 3:  # Need at least 3 points for a polygon
                continue
                
            region_landmarks = landmarks[indices].astype(np.int32)
            color = self.region_colors[region_name]
            
            # Create convex hull for the region
            if region_name in ['right_eye', 'left_eye', 'mouth']:
                # For eyes and mouth, use the exact points
                cv2.fillPoly(overlay, [region_landmarks], color)
            else:
                # For other regions, use convex hull
                hull = cv2.convexHull(region_landmarks)
                cv2.fillPoly(overlay, [hull], color)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return result
    
    def plot_landmarks_matplotlib(self, image, landmarks, save_path=None):
        """
        Create a matplotlib plot of the image with landmarks.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Plot landmarks for each region with different colors
        for region_name, indices in self.landmark_regions.items():
            region_landmarks = landmarks[indices]
            color = [c/255.0 for c in self.region_colors[region_name][::-1]]  # Convert BGR to RGB and normalize
            
            ax.scatter(region_landmarks[:, 0], region_landmarks[:, 1], 
                      c=[color], s=20, label=region_name.replace('_', ' ').title())
        
        ax.set_title('Facial Landmark Detection', fontsize=16)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_landmark_grid(self, images_and_landmarks, grid_size=(2, 2), figsize=(15, 10)):
        """
        Create a grid of images with landmarks for comparison.
        
        Args:
            images_and_landmarks (list): List of tuples (image, landmarks)
            grid_size (tuple): Grid dimensions (rows, cols)
            figsize (tuple): Figure size for matplotlib
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (image, landmarks) in enumerate(images_and_landmarks[:len(axes)]):
            if i >= len(axes):
                break
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(image_rgb)
            
            # Plot landmarks
            for region_name, indices in self.landmark_regions.items():
                region_landmarks = landmarks[indices]
                color = [c/255.0 for c in self.region_colors[region_name][::-1]]
                axes[i].scatter(region_landmarks[:, 0], region_landmarks[:, 1], 
                              c=[color], s=15)
            
            axes[i].set_title(f'Image {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images_and_landmarks), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def draw_landmark_numbers(self, image, landmarks, font_scale=0.3, color=(255, 255, 255)):
        """
        Draw landmark point numbers on the image.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Array of landmark coordinates
            font_scale (float): Font scale for the numbers
            color (tuple): BGR color for the text
            
        Returns:
            numpy.ndarray: Image with landmark numbers drawn
        """
        image_copy = image.copy()
        
        for i, (x, y) in enumerate(landmarks):
            cv2.putText(image_copy, str(i), (int(x) + 2, int(y) - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        return image_copy
    
    def create_landmark_analysis_plot(self, landmarks, measurements=None):
        """
        Create an analysis plot showing facial measurements and proportions.
        
        Args:
            landmarks (numpy.ndarray): Array of landmark coordinates
            measurements (dict, optional): Dictionary of facial measurements
            
        Returns:
            matplotlib.figure.Figure: The created analysis figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Landmark positions with measurements
        ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20)
        
        # Draw measurement lines if provided
        if measurements:
            # Inter-ocular distance
            left_eye_center = measurements.get('left_eye_center', [0, 0])
            right_eye_center = measurements.get('right_eye_center', [0, 0])
            ax1.plot([left_eye_center[0], right_eye_center[0]], 
                    [left_eye_center[1], right_eye_center[1]], 
                    'b-', linewidth=2, label='Inter-ocular distance')
        
        ax1.set_aspect('equal')
        ax1.invert_yaxis()  # Flip y-axis to match image coordinates
        ax1.set_title('Facial Landmarks with Measurements')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Measurements bar chart
        if measurements:
            metrics = {k: v for k, v in measurements.items() 
                      if isinstance(v, (int, float)) and k != 'face_ratio'}
            
            names = list(metrics.keys())
            values = list(metrics.values())
            
            ax2.barh(names, values, color='skyblue')
            ax2.set_xlabel('Pixels')
            ax2.set_title('Facial Measurements')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax2.text(v + max(values) * 0.01, i, f'{v:.1f}', 
                        va='center', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No measurements provided', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Facial Measurements')
        
        plt.tight_layout()
        return fig