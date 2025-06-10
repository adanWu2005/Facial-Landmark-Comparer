#!/usr/bin/env python3
"""
Download script for dlib facial landmark detection models.

This script downloads the pre-trained dlib facial landmark predictor models
and places them in the correct directory structure.
"""

import os
import sys
import urllib.request
import bz2
import shutil
from pathlib import Path
import hashlib


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "models",
        "images/input",
        "images/output",
        "config",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def download_file(url, filename, expected_hash=None):
    """
    Download a file from URL with progress indication.
    
    Args:
        url (str): URL to download from
        filename (str): Local filename to save to
        expected_hash (str): Expected SHA256 hash for verification
    """
    print(f"Downloading {filename}...")
    
    def show_progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\rProgress: {percent}% ({block_num * block_size}/{total_size} bytes)", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print(f"\nDownload completed: {filename}")
        
        # Verify file hash if provided
        if expected_hash:
            if verify_file_hash(filename, expected_hash):
                print("File hash verification: PASSED")
            else:
                print("File hash verification: FAILED")
                return False
                
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False


def verify_file_hash(filename, expected_hash):
    """
    Verify SHA256 hash of downloaded file.
    
    Args:
        filename (str): Path to file
        expected_hash (str): Expected SHA256 hash
        
    Returns:
        bool: True if hash matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest().lower() == expected_hash.lower()


def extract_bz2(compressed_file, output_file):
    """
    Extract a bz2 compressed file.
    
    Args:
        compressed_file (str): Path to compressed file
        output_file (str): Path for extracted file
    """
    print(f"Extracting {compressed_file}...")
    
    try:
        with bz2.BZ2File(compressed_file, 'rb') as source:
            with open(output_file, 'wb') as target:
                shutil.copyfileobj(source, target)
        
        print(f"Extraction completed: {output_file}")
        
        # Remove compressed file
        os.remove(compressed_file)
        print(f"Removed compressed file: {compressed_file}")
        
        return True
    except Exception as e:
        print(f"Error extracting {compressed_file}: {e}")
        return False


def download_68_point_model():
    """Download the 68-point facial landmark predictor model."""
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "models/shape_predictor_68_face_landmarks.dat.bz2"
    extracted_file = "models/shape_predictor_68_face_landmarks.dat"
    
    # Expected hash for verification (this is an example - you should verify the actual hash)
    # expected_hash = "fbdc2cb60428449b8bc92f49c8d0f564568b06ba8a8c3d90e556d1aac10b43ef"
    
    if os.path.exists(extracted_file):
        print(f"Model already exists: {extracted_file}")
        return True
    
    # Download compressed model
    if download_file(model_url, compressed_file):
        # Extract the model
        if extract_bz2(compressed_file, extracted_file):
            print("68-point facial landmark model ready!")
            return True
    
    return False


def download_5_point_model():
    """Download the 5-point facial landmark predictor model (optional)."""
    model_url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
    compressed_file = "models/shape_predictor_5_face_landmarks.dat.bz2"
    extracted_file = "models/shape_predictor_5_face_landmarks.dat"
    
    if os.path.exists(extracted_file):
        print(f"Model already exists: {extracted_file}")
        return True
    
    print("Downloading optional 5-point model...")
    
    # Download compressed model
    if download_file(model_url, compressed_file):
        # Extract the model
        if extract_bz2(compressed_file, extracted_file):
            print("5-point facial landmark model ready!")
            return True
    
    return False


def create_sample_image():
    """Create a simple sample image for testing."""
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add some text
        cv2.putText(img, "Place your test images", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "in images/input/", (80, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "directory", (140, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save sample image
        cv2.imwrite("images/input/sample_placeholder.jpg", img)
        print("Created sample placeholder image: images/input/sample_placeholder.jpg")
        
    except ImportError:
        print("OpenCV not installed. Skipping sample image creation.")
    except Exception as e:
        print(f"Error creating sample image: {e}")


def create_init_files():
    """Create __init__.py files for Python packages."""
    init_files = [
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        Path(init_file).touch(exist_ok=True)
        print(f"Created: {init_file}")


def verify_installation():
    """Verify that required packages are installed."""
    required_packages = ["cv2", "dlib", "numpy", "yaml"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("\nAll required packages are installed!")
    return True


def main():
    """Main function to set up the facial landmark detection project."""
    print("Setting up OpenCV Facial Landmark Detection Project")
    print("=" * 50)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directories()
    
    # Create __init__.py files
    print("\n2. Creating package files...")
    create_init_files()
    
    # Verify package installation
    print("\n3. Verifying package installation...")
    if not verify_installation():
        print("\nSetup incomplete. Please install required packages first.")
        sys.exit(1)
    
    # Download models
    print("\n4. Downloading facial landmark models...")
    
    # Download 68-point model (required)
    if not download_68_point_model():
        print("Failed to download 68-point model. Please download manually from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)
    
    # Download 5-point model (optional)
    user_input = input("\nDownload 5-point model as well? (y/N): ").lower().strip()
    if user_input in ['y', 'yes']:
        download_5_point_model()
    
    # Create sample files
    print("\n5. Creating sample files...")
    create_sample_image()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nTo test the installation, run:")
    print("python src/facial_landmarks.py --image images/input/your_image.jpg")
    print("\nFor help, run:")
    print("python src/facial_landmarks.py --help")


if __name__ == "__main__":
    main()