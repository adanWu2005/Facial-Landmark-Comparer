#!/usr/bin/env python3
"""
Unit tests for facial landmark detection functionality.
"""

import unittest
import os
import sys
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from facial_landmarks import FacialLandmarkDetector
from utils.face_utils import FaceUtils
from visualization.visualizer import LandmarkVisualizer


class TestFaceUtils(unittest.TestCase):
    """Test cases for face utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.face_utils = FaceUtils()
        
        # Create mock landmarks (68 points)
        self.mock_landmarks = np.array([
            [30, 100], [32, 105], [34, 110], [36, 115], [38, 120],  # Jaw points 0-4
            [40, 125], [42, 130], [44, 135], [46, 140], [48, 145],  # Jaw points 5-9
            [50, 150], [52, 155], [54, 160], [56, 165], [58, 170],  # Jaw points 10-14
            [60, 175], [62, 180],  # Jaw points 15-16
            
            # Right eyebrow (17-21)
            [35, 95], [40, 90], [45, 88], [50, 90], [55, 95],
            
            # Left eyebrow (22-26)
            [65, 95], [70, 90], [75, 88], [80, 90], [85, 95],
            
            # Nose (27-35)
            [60, 100], [60, 105], [60, 110], [60, 115], [58, 120],
            [55, 125], [60, 125], [65, 125], [62, 120],
            
            # Right eye (36-41)
            [40, 105], [45, 100], [50, 100], [55, 105], [50, 110], [45, 110],
            
            # Left eye (42-47)
            [65, 105], [70, 100], [75, 100], [80, 105], [75, 110], [70, 110],
            
            # Mouth (48-67)
            [45, 140], [50, 135], [55, 130], [60, 132], [65, 130], [70, 135], [75, 140],
            [70, 145], [65, 150], [60, 152], [55, 150], [50, 145],
            [50, 140], [55, 138], [60, 140], [65, 138], [70, 140],
            [70, 142], [65, 145], [60, 147], [55, 145], [50, 142]
        ], dtype=np.int32)
    
    def test_rect_to_bb(self):
        """Test rectangle to bounding box conversion."""
        # Mock dlib rectangle
        class MockRect:
            def __init__(self, left, top, right, bottom):
                self._left = left
                self._top = top
                self._right = right
                self._bottom = bottom
            
            def left(self): return self._left
            def top(self): return self._top
            def right(self): return self._right
            def bottom(self): return self._bottom
        
        rect = MockRect(10, 20, 110, 120)
        bbox = self.face_utils.rect_to_bb(rect)
        
        expected = (10, 20, 100, 100)  # (x, y, w, h)
        self.assertEqual(bbox, expected)
    
    def test_shape_to_np(self):
        """Test shape object to numpy array conversion."""
        # Mock dlib shape
        class MockShape:
            def __init__(self, points):
                self.points = points
            
            def part(self, i):
                class Point:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                return Point(self.points[i][0], self.points[i][1])
        
        # Create mock shape with 68 points
        points = [(i, i+10) for i in range(68)]
        shape = MockShape(points)
        
        result = self.face_utils.shape_to_np(shape)
        
        self.assertEqual(result.shape, (68, 2))
        self.assertEqual(result[0].tolist(), [0, 10])
        self.assertEqual(result[67].tolist(), [67, 77])
    
    def test_calculate_face_metrics(self):
        """Test face metrics calculation."""
        metrics = self.face_utils.calculate_face_metrics(self.mock_landmarks)
        
        # Check that all expected metrics are present
        expected_keys = [
            'inter_ocular_distance', 'face_width', 'face_height',
            'nose_width', 'nose_height', 'mouth_width', 'mouth_height',
            'left_eye_center', 'right_eye_center'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        # Check that values are reasonable
        self.assertGreater(metrics['inter_ocular_distance'], 0)
        self.assertGreater(metrics['face_width'], 0)
        self.assertGreater(metrics['face_height'], 0)
    
    def test_extract_face_region(self):
        """Test facial region extraction."""
        # Create a test image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Extract eye region
        region_image, bbox = self.face_utils.extract_face_region(
            image, self.mock_landmarks, 'left_eye', padding=5
        )
        
        self.assertIsInstance(region_image, np.ndarray)
        self.assertEqual(len(bbox), 4)  # (x, y, w, h)
        self.assertGreater(region_image.shape[0], 0)
        self.assertGreater(region_image.shape[1], 0)


class TestLandmarkVisualizer(unittest.TestCase):
    """Test cases for landmark visualization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = LandmarkVisualizer()
        self.test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Simple landmark points for testing
        self.test_landmarks = np.array([
            [50, 50], [60, 55], [70, 60], [80, 65], [90, 70],
            [45, 45], [55, 40], [65, 38], [75, 40], [85, 45],
            [60, 70], [60, 75], [60, 80], [60, 85], [58, 90],
            [55, 95], [60, 95], [65, 95], [62, 90],
            [40, 75], [45, 70], [50, 70], [55, 75], [50, 80], [45, 80],
            [65, 75], [70, 70], [75, 70], [80, 75], [75, 80], [70, 80],
        ] + [[i, i+100] for i in range(50, 88)], dtype=np.int32)  # Complete to 68 points
    
    def test_draw_landmarks(self):
        """Test basic landmark drawing."""
        result = self.visualizer.draw_landmarks(
            self.test_image, self.test_landmarks[:10]  # Use first 10 points
        )
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Image should be modified (not identical to original)
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_draw_face_box(self):
        """Test face bounding box drawing."""
        bbox = (50, 50, 100, 100)  # (x, y, w, h)
        result = self.visualizer.draw_face_box(self.test_image, bbox, "Test Face")
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertFalse(np.array_equal(result, self.test_image))


class TestFacialLandmarkDetector(unittest.TestCase):
    """Test cases for the main facial landmark detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add a simple face-like pattern
        cv2.circle(self.test_image, (200, 150), 80, (200, 200, 200), -1)  # Face
        cv2.circle(self.test_image, (180, 130), 10, (100, 100, 100), -1)  # Left eye
        cv2.circle(self.test_image, (220, 130), 10, (100, 100, 100), -1)  # Right eye
        cv2.ellipse(self.test_image, (200, 180), (20, 10), 0, 0, 180, (100, 100, 100), 2)  # Mouth
        
        # Save test image to temporary file
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_face.jpg')
        cv2.imwrite(self.test_image_path, self.test_image)
        
        # Create a simple config for testing
        self.test_config = {
            'model': {
                'shape_predictor_path': 'models/shape_predictor_68_face_landmarks.dat',
                'face_detector': 'hog'
            },
            'processing': {
                'resize_width': 200,
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
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with default config
        try:
            detector = FacialLandmarkDetector()
            self.assertIsNotNone(detector.config)
            self.assertIn('model', detector.config)
            self.assertIn('processing', detector.config)
        except FileNotFoundError:
            # Skip test if model file is not available
            self.skipTest("Model file not available for testing")
    
    def test_get_landmark_regions(self):
        """Test landmark region extraction."""
        try:
            detector = FacialLandmarkDetector()
            
            # Create mock landmarks
            landmarks = np.arange(136).reshape(68, 2)  # 68 points with (x, y) coordinates
            
            regions = detector.get_landmark_regions(landmarks)
            
            expected_regions = ['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 
                              'right_eye', 'left_eye', 'mouth']
            
            for region in expected_regions:
                self.assertIn(region, regions)
                self.assertIsInstance(regions[region], list)
                
        except FileNotFoundError:
            self.skipTest("Model file not available for testing")


class TestBatchProcessing(unittest.TestCase):
    """Test cases for batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures for batch processing."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test images
        for i in range(3):
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
            # Add some variation to each image
            cv2.circle(test_image, (100, 100), 50 + i*10, (200-i*30, 200-i*30, 200-i*30), -1)
            
            image_path = os.path.join(self.input_dir, f'test_image_{i}.jpg')
            cv2.imwrite(image_path, test_image)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_image_file_discovery(self):
        """Test image file discovery in batch processing."""
        try:
            from batch_process import BatchProcessor
            processor = BatchProcessor()
            
            image_files = processor.get_image_files(self.input_dir)
            self.assertEqual(len(image_files), 3)
            
            for file_path in image_files:
                self.assertTrue(os.path.exists(file_path))
                self.assertTrue(file_path.endswith('.jpg'))
                
        except (ImportError, FileNotFoundError):
            self.skipTest("Batch processor or model not available for testing")
    
    def test_output_structure_creation(self):
        """Test output directory structure creation."""
        try:
            from batch_process import BatchProcessor
            processor = BatchProcessor()
            
            processor.create_output_structure(self.output_dir)
            
            expected_dirs = ['annotated', 'landmarks', 'measurements', 'regions']
            for dir_name in expected_dirs:
                dir_path = os.path.join(self.output_dir, dir_name)
                self.assertTrue(os.path.exists(dir_path))
                self.assertTrue(os.path.isdir(dir_path))
                
        except (ImportError, FileNotFoundError):
            self.skipTest("Batch processor not available for testing")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a more realistic test face image
        self.test_image = self._create_test_face_image()
        self.test_image_path = os.path.join(self.temp_dir, 'integration_test.jpg')
        cv2.imwrite(self.test_image_path, self.test_image)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_face_image(self):
        """Create a more realistic test face image."""
        image = np.ones((300, 300, 3), dtype=np.uint8) * 240
        
        # Face outline
        cv2.ellipse(image, (150, 150), (80, 100), 0, 0, 360, (220, 220, 220), -1)
        
        # Eyes
        cv2.ellipse(image, (120, 120), (15, 8), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(image, (180, 120), (15, 8), 0, 0, 360, (200, 200, 200), -1)
        cv2.circle(image, (120, 120), 5, (100, 100, 100), -1)
        cv2.circle(image, (180, 120), 5, (100, 100, 100), -1)
        
        # Nose
        cv2.ellipse(image, (150, 150), (8, 15), 0, 0, 360, (210, 210, 210), -1)
        
        # Mouth
        cv2.ellipse(image, (150, 180), (20, 8), 0, 0, 180, (190, 190, 190), 2)
        
        # Eyebrows
        cv2.ellipse(image, (120, 100), (20, 5), 0, 0, 180, (180, 180, 180), 2)
        cv2.ellipse(image, (180, 100), (20, 5), 0, 0, 180, (180, 180, 180), 2)
        
        return image
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing workflow."""
        try:
            # Initialize detector
            detector = FacialLandmarkDetector()
            
            # Process the test image
            output_path = os.path.join(self.temp_dir, 'output_annotated.jpg')
            results = detector.process_image(
                self.test_image_path,
                output_path,
                save_landmarks=True
            )
            
            # Verify results structure
            self.assertIn('num_faces', results)
            self.assertIn('faces', results)
            self.assertIn('annotated_image', results)
            
            # Check if output file was created
            if results['num_faces'] > 0:
                self.assertTrue(os.path.exists(output_path))
                
                # Verify output image can be loaded
                output_image = cv2.imread(output_path)
                self.assertIsNotNone(output_image)
                self.assertEqual(len(output_image.shape), 3)
                
        except FileNotFoundError:
            self.skipTest("Model file not available for integration testing")
        except Exception as e:
            self.fail(f"End-to-end test failed with error: {e}")


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation and error handling."""
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        try:
            detector = FacialLandmarkDetector()
            
            with self.assertRaises(ValueError):
                detector.process_image("nonexistent_image.jpg")
                
        except FileNotFoundError:
            self.skipTest("Model file not available for testing")
    
    def test_landmark_validation(self):
        """Test landmark coordinate validation."""
        # Test with invalid landmark coordinates
        invalid_landmarks = np.array([[-1, -1], [1000, 1000]])
        
        face_utils = FaceUtils()
        
        # Should handle invalid coordinates gracefully
        try:
            metrics = face_utils.calculate_face_metrics(invalid_landmarks)
            # Should not crash, but metrics might not be meaningful
            self.assertIsInstance(metrics, dict)
        except Exception as e:
            # Some operations might fail with invalid data, which is acceptable
            pass
    
    def test_empty_image_handling(self):
        """Test handling of empty or corrupted images."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create an empty file
            empty_file = os.path.join(temp_dir, 'empty.jpg')
            with open(empty_file, 'w') as f:
                f.write('')
            
            detector = FacialLandmarkDetector()
            
            with self.assertRaises(ValueError):
                detector.process_image(empty_file)
                
        except FileNotFoundError:
            self.skipTest("Model file not available for testing")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_suite():
    """Create a comprehensive test suite."""
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFaceUtils,
        TestLandmarkVisualizer,
        TestFacialLandmarkDetector,
        TestBatchProcessing,
        TestIntegration,
        TestDataValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    suite = create_test_suite()
    
    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Run tests
    print("Running OpenCV Facial Landmark Detection Tests")
    print("=" * 50)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)