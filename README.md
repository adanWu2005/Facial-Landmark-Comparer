# OpenCSV Facial Landmark

Facial landmark detection using dlib and OpenCV, with a modern Gradio web interface for easy use.

## Features
- Detects 68 facial landmarks on faces in images
- Upload two photos or use your webcam for detection
- Renders facial landmarks and their indices on the original image
- Also renders landmarks (with numbers) on a white background for each image
- Simple, modern web interface (Gradio)
- Supports image upload and webcam capture

## Demo
![Demo Screenshot](demo_screenshot.png)

## Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd OpenCSV-Facial-Landmark
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dlib 68-point shape predictor model:**
   - Download from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
   - Extract the `.bz2` file and place `shape_predictor_68_face_landmarks.dat` in the `models/` directory.

## Usage
### Web Interface
Run the Gradio app:
```bash
python src/app.py
```
- Open the link shown in your terminal (usually http://127.0.0.1:7860/)
- Upload two images or use your webcam
- View the results: original with landmarks, and landmarks on white background

### Command Line
You can also run detection on a single image:
```bash
python src/facial_landmarks.py --image path/to/image.jpg
```

## Notes
- **AVIF images are NOT supported.**
- The model file (`shape_predictor_68_face_landmarks.dat`) is required in the `models/` directory.
- Results are saved in the same directory as the input image with suffixes.

## File Structure
- `src/app.py` — Gradio web interface
- `src/facial_landmarks.py` — Command-line facial landmark detection
- `src/utils/face_utils.py` — Utility functions for face/landmark processing
- `src/visualization/visualizer.py` — Drawing and visualization utilities
- `models/` — Place your dlib model file here

## Credits
- Built with [dlib](http://dlib.net/), [OpenCV](https://opencv.org/), and [Gradio](https://gradio.app/)
- 68-point model from [dlib-models](https://github.com/davisking/dlib-models)

