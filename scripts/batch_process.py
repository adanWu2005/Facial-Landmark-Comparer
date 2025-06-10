#!/usr/bin/env python3
"""
Batch processing script for facial landmark detection.

This script processes multiple images in a directory and generates
facial landmark annotations for all detected faces.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from facial_landmarks import FacialLandmarkDetector


class BatchProcessor:
    """
    Class for batch processing multiple images for facial landmark detection.
    """
    
    def __init__(self, config_path=None, num_threads=None):
        """
        Initialize the batch processor.
        
        Args:
            config_path (str): Path to configuration file
            num_threads (int): Number of threads for parallel processing
        """
        self.detector = FacialLandmarkDetector(config_path)
        self.num_threads = num_threads or os.cpu_count()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def get_image_files(self, input_dir):
        """
        Get all supported image files from the input directory.
        
        Args:
            input_dir (str): Input directory path
            
        Returns:
            list: List of image file paths
        """
        image_files = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def create_output_structure(self, output_dir):
        """
        Create output directory structure for batch processing.
        
        Args:
            output_dir (str): Output directory path
        """
        output_path = Path(output_dir)
        subdirs = ['annotated', 'landmarks', 'measurements', 'regions']
        
        for subdir in subdirs:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def process_single_image(self, image_path, output_dir, save_landmarks=True, 
                           save_measurements=True, extract_regions=False):
        """
        Process a single image for facial landmark detection.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Output directory
            save_landmarks (bool): Whether to save landmarks as JSON
            save_measurements (bool): Whether to save facial measurements
            extract_regions (bool): Whether to extract facial regions
            
        Returns:
            dict: Processing results
        """
        try:
            start_time = time.time()
            
            # Get base filename without extension
            base_name = Path(image_path).stem
            
            # Define output paths
            annotated_path = os.path.join(output_dir, 'annotated', f"{base_name}_annotated.jpg")
            landmarks_path = os.path.join(output_dir, 'landmarks', f"{base_name}_landmarks.json")
            measurements_path = os.path.join(output_dir, 'measurements', f"{base_name}_measurements.json")
            
            # Process image
            results = self.detector.process_image(
                image_path, 
                annotated_path, 
                save_landmarks=False  # We'll handle saving ourselves
            )
            
            processing_time = time.time() - start_time
            
            # Save landmarks if requested
            if save_landmarks and results['num_faces'] > 0:
                with open(landmarks_path, 'w') as f:
                    json.dump({
                        'image_path': image_path,
                        'processing_time': processing_time,
                        'faces': results['faces']
                    }, f, indent=2)
            
            # Calculate and save measurements if requested
            if save_measurements and results['num_faces'] > 0:
                from utils.face_utils import FaceUtils
                
                measurements_data = {
                    'image_path': image_path,
                    'processing_time': processing_time,
                    'faces': []
                }
                
                for face in results['faces']:
                    landmarks = face['landmarks']
                    measurements = FaceUtils.calculate_face_metrics(landmarks)
                    analysis = FaceUtils.detect_facial_features(landmarks)
                    
                    face_measurements = {
                        'face_id': face['face_id'],
                        'bbox': face['bbox'],
                        'measurements': measurements,
                        'analysis': analysis
                    }
                    measurements_data['faces'].append(face_measurements)
                
                with open(measurements_path, 'w') as f:
                    json.dump(measurements_data, f, indent=2)
            
            # Extract facial regions if requested
            if extract_regions and results['num_faces'] > 0:
                self._extract_facial_regions(image_path, results, output_dir)
            
            return {
                'image_path': image_path,
                'success': True,
                'num_faces': results['num_faces'],
                'processing_time': processing_time,
                'output_files': {
                    'annotated': annotated_path if results['num_faces'] > 0 else None,
                    'landmarks': landmarks_path if save_landmarks and results['num_faces'] > 0 else None,
                    'measurements': measurements_path if save_measurements and results['num_faces'] > 0 else None
                }
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'num_faces': 0,
                'processing_time': 0
            }
    
    def _extract_facial_regions(self, image_path, results, output_dir):
        """
        Extract facial regions from detected faces.
        
        Args:
            image_path (str): Path to original image
            results (dict): Detection results
            output_dir (str): Output directory
        """
        from utils.face_utils import FaceUtils
        
        image = cv2.imread(image_path)
        base_name = Path(image_path).stem
        regions_dir = os.path.join(output_dir, 'regions', base_name)
        Path(regions_dir).mkdir(parents=True, exist_ok=True)
        
        regions_to_extract = ['left_eye', 'right_eye', 'nose', 'mouth']
        
        for face_idx, face in enumerate(results['faces']):
            landmarks = face['landmarks']
            
            for region_name in regions_to_extract:
                try:
                    region_image, bbox = FaceUtils.extract_face_region(
                        image, landmarks, region_name, padding=10
                    )
                    
                    region_filename = f"face_{face_idx + 1}_{region_name}.jpg"
                    region_path = os.path.join(regions_dir, region_filename)
                    cv2.imwrite(region_path, region_image)
                    
                except Exception as e:
                    print(f"Warning: Could not extract {region_name} from face {face_idx + 1}: {e}")
    
    def process_batch(self, input_dir, output_dir, save_landmarks=True, 
                     save_measurements=True, extract_regions=False, 
                     parallel=True, show_progress=True):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Input directory containing images
            output_dir (str): Output directory for results
            save_landmarks (bool): Whether to save landmarks as JSON
            save_measurements (bool): Whether to save facial measurements
            extract_regions (bool): Whether to extract facial regions
            parallel (bool): Whether to use parallel processing
            show_progress (bool): Whether to show progress updates
            
        Returns:
            dict: Batch processing results
        """
        # Get list of image files
        image_files = self.get_image_files(input_dir)
        
        if not image_files:
            raise ValueError(f"No supported image files found in: {input_dir}")
        
        print(f"Found {len(image_files)} images to process")
        
        # Create output directory structure
        self.create_output_structure(output_dir)
        
        # Initialize results
        batch_results = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'total_images': len(image_files),
            'processed_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_faces': 0,
            'total_processing_time': 0,
            'results': []
        }
        
        start_time = time.time()
        
        if parallel and len(image_files) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.process_single_image, 
                        img_path, output_dir, save_landmarks, 
                        save_measurements, extract_regions
                    ): img_path for img_path in image_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_image):
                    result = future.result()
                    batch_results['results'].append(result)
                    batch_results['processed_images'] += 1
                    
                    if result['success']:
                        batch_results['successful_images'] += 1
                        batch_results['total_faces'] += result['num_faces']
                    else:
                        batch_results['failed_images'] += 1
                        if show_progress:
                            print(f"Failed: {result['image_path']} - {result['error']}")
                    
                    batch_results['total_processing_time'] += result['processing_time']
                    
                    if show_progress:
                        progress = (batch_results['processed_images'] / len(image_files)) * 100
                        print(f"Progress: {progress:.1f}% ({batch_results['processed_images']}/{len(image_files)}) - "
                              f"Faces found: {batch_results['total_faces']}")
        else:
            # Sequential processing
            for i, image_path in enumerate(image_files):
                result = self.process_single_image(
                    image_path, output_dir, save_landmarks, 
                    save_measurements, extract_regions
                )
                
                batch_results['results'].append(result)
                batch_results['processed_images'] += 1
                
                if result['success']:
                    batch_results['successful_images'] += 1
                    batch_results['total_faces'] += result['num_faces']
                else:
                    batch_results['failed_images'] += 1
                    if show_progress:
                        print(f"Failed: {result['image_path']} - {result['error']}")
                
                batch_results['total_processing_time'] += result['processing_time']
                
                if show_progress:
                    progress = ((i + 1) / len(image_files)) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(image_files)}) - "
                          f"Faces found: {batch_results['total_faces']}")
        
        batch_results['wall_clock_time'] = time.time() - start_time
        
        # Save batch results
        results_path = os.path.join(output_dir, 'batch_results.json')
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        return batch_results
    
    def generate_summary_report(self, batch_results, output_path=None):
        """
        Generate a summary report of batch processing results.
        
        Args:
            batch_results (dict): Results from batch processing
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Summary report text
        """
        report_lines = [
            "Facial Landmark Detection - Batch Processing Summary",
            "=" * 55,
            "",
            f"Input Directory: {batch_results['input_dir']}",
            f"Output Directory: {batch_results['output_dir']}",
            f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Results Overview:",
            f"  Total Images: {batch_results['total_images']}",
            f"  Successfully Processed: {batch_results['successful_images']}",
            f"  Failed: {batch_results['failed_images']}",
            f"  Total Faces Detected: {batch_results['total_faces']}",
            "",
            "Performance Metrics:",
            f"  Total Processing Time: {batch_results['total_processing_time']:.2f} seconds",
            f"  Wall Clock Time: {batch_results['wall_clock_time']:.2f} seconds",
            f"  Average Time per Image: {batch_results['total_processing_time'] / max(1, batch_results['processed_images']):.2f} seconds",
            f"  Average Faces per Image: {batch_results['total_faces'] / max(1, batch_results['successful_images']):.2f}",
            "",
            "Detailed Results:"
        ]
        
        # Add details for each processed image
        for result in batch_results['results']:
            if result['success']:
                report_lines.append(
                    f"  ✓ {Path(result['image_path']).name}: "
                    f"{result['num_faces']} faces ({result['processing_time']:.2f}s)"
                )
            else:
                report_lines.append(
                    f"  ✗ {Path(result['image_path']).name}: "
                    f"FAILED - {result['error']}"
                )
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {output_path}")
        
        return report_text


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch process images for facial landmark detection')
    parser.add_argument('-i', '--input-dir', required=True,
                      help='Input directory containing images')
    parser.add_argument('-o', '--output-dir', required=True,
                      help='Output directory for results')
    parser.add_argument('-c', '--config',
                      help='Path to configuration file')
    parser.add_argument('--no-landmarks', action='store_true',
                      help='Do not save landmarks JSON files')
    parser.add_argument('--no-measurements', action='store_true',
                      help='Do not save facial measurements')
    parser.add_argument('--extract-regions', action='store_true',
                      help='Extract facial regions (eyes, nose, mouth)')
    parser.add_argument('--sequential', action='store_true',
                      help='Process images sequentially (no parallel processing)')
    parser.add_argument('--threads', type=int,
                      help='Number of threads for parallel processing')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress progress output')
    parser.add_argument('--report', 
                      help='Path to save summary report')
    
    args = parser.parse_args()
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(config_path=args.config, num_threads=args.threads)
        
        print("Starting batch processing...")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        
        # Process batch
        results = processor.process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_landmarks=not args.no_landmarks,
            save_measurements=not args.no_measurements,
            extract_regions=args.extract_regions,
            parallel=not args.sequential,
            show_progress=not args.quiet
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("Batch Processing Complete!")
        print(f"Processed: {results['successful_images']}/{results['total_images']} images")
        print(f"Total faces detected: {results['total_faces']}")
        print(f"Processing time: {results['total_processing_time']:.2f} seconds")
        print(f"Wall clock time: {results['wall_clock_time']:.2f} seconds")
        
        if results['failed_images'] > 0:
            print(f"Failed images: {results['failed_images']}")
        
        # Generate and save report
        report_path = args.report or os.path.join(args.output_dir, 'summary_report.txt')
        report = processor.generate_summary_report(results, report_path)
        
        if not args.quiet:
            print(f"\nResults saved to: {args.output_dir}")
            print(f"Summary report: {report_path}")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()