import gradio as gr
import cv2
import numpy as np
from facial_landmarks import FacialLandmarkDetector

# Initialize detector
facial_detector = FacialLandmarkDetector()

def process_image(image):
    if image is None:
        return None, None
    # Convert to BGR if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Resize for consistency
    h, w = image.shape[:2]
    if w > 800:
        scale = 800 / w
        image = cv2.resize(image, (800, int(h * scale)))
    faces = facial_detector.detect_faces(image)
    annotated = image.copy()
    white_img = 255 * np.ones_like(image)
    for i, face_rect in enumerate(faces):
        landmarks = facial_detector.detect_landmarks(image, face_rect)
        bbox = facial_detector.face_utils.rect_to_bb(face_rect)
        annotated = facial_detector.visualizer.draw_face_box(annotated, bbox, f"Face #{i+1}")
        annotated = facial_detector.visualizer.draw_landmarks(annotated, landmarks)
        annotated = facial_detector.visualizer.draw_landmark_numbers(annotated, landmarks)
        white_img = facial_detector.visualizer.draw_landmarks(white_img, landmarks)
        white_img = facial_detector.visualizer.draw_landmark_numbers(white_img, landmarks, color=(0,0,0))
    return annotated, white_img

def interface(img1, img2, webcam_img):
    out1, white1 = process_image(img1) if img1 is not None else (None, None)
    out2, white2 = process_image(img2) if img2 is not None else (None, None)
    out3, white3 = process_image(webcam_img) if webcam_img is not None else (None, None)
    return out1, white1, out2, white2, out3, white3

with gr.Blocks() as demo:
    gr.Markdown("""
    # Facial Landmark Detection
    Upload two photos or use your webcam. For each, you'll see:
    - The original with facial landmarks and numbers
    - The landmarks on a white background
    
    AVIF images are NOT supported.
    """)
    with gr.Row():
        img1 = gr.Image(label="Photo 1", type="numpy", sources=["upload", "webcam"])
        img2 = gr.Image(label="Photo 2", type="numpy", sources=["upload", "webcam"])
    btn = gr.Button("Process")
    with gr.Row():
        out1 = gr.Image(label="Photo 1 with Landmarks")
        white1 = gr.Image(label="Photo 1 Landmarks on White")
        out2 = gr.Image(label="Photo 2 with Landmarks")
        white2 = gr.Image(label="Photo 2 Landmarks on White")
    btn.click(lambda img1, img2: interface(img1, img2, None), inputs=[img1, img2], outputs=[out1, white1, out2, white2])

demo.launch()  