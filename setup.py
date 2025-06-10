#!/usr/bin/env python3
"""
Setup script for OpenCV Facial Landmark Detection package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "OpenCV Facial Landmark Detection Project"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="opencv-facial-landmarks",
    version="1.0.0",
    description="Facial landmark detection using dlib and OpenCV",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/opencv-facial-landmarks",
    license="MIT",
    
    # Package information
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
        "gui": [
            "tkinter>=8.6.0",
            "PyQt5>=5.15.0",
        ],
        "performance": [
            "numba>=0.53.0",
            "opencv-contrib-python>=4.5.0",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "facial-landmarks=facial_landmarks:main",
            "batch-process=batch_process:main",
            "download-models=download_model:main",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
    ],
    
    # Keywords
    keywords="facial landmarks, computer vision, opencv, dlib, face detection, image processing",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/opencv-facial-landmarks/issues",
        "Source": "https://github.com/yourusername/opencv-facial-landmarks",
        "Documentation": "https://github.com/yourusername/opencv-facial-landmarks/wiki",
    },
)