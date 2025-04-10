import cv2
import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import csv
from datetime import datetime
import threading
import argparse
import time
import tkinter as tk
from tkinter import ttk

# Global variables
text_prompts = []
lock = threading.Lock()
last_detections = []

# A simple Tkinter dashboard to display detected objects in a list.

class SimpleDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Dashboard")
        self.root.geometry("300x200")

        self.detection_label = ttk.Label(root, text="Detected Objects:", font=('Helvetica', 12))
        self.detection_label.pack(pady=10)

        self.objects_list = tk.Listbox(root, height=5, font=('Helvetica', 10))
        self.objects_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    #Updates the detected objects displayed in the listbox.
    def update_detections(self, detections):

        self.objects_list.delete(0, tk.END)
        for label, score in detections:
            self.objects_list.insert(tk.END, f"{label}: {score:.2f}")

# Function to set up the CSV logger
def setup_logger(log_file='detections.csv'):

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'frame_number', 'label', 'score', 'x1', 'y1', 'x2', 'y2'])
    return log_file

# Function to log detections in the CSV file
def log_detection(log_file, frame_number, label, score, box):


    timestamp = datetime.now().isoformat()
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, frame_number, label, score, box[0], box[1], box[2], box[3]])

# Function to update object detection prompts
def update_prompts():

    global text_prompts
    while True:
        new_prompts = input("Enter new object categories (comma-separated): ").strip()
        if new_prompts:
            with lock:
                text_prompts = [f"a {p.strip()}" for p in new_prompts.split(',')]
            print(f"Updated prompts to: {text_prompts}")

# Function to process a video frame for object detection
def process_frame(frame, processor, model, device, confidence_threshold=0.3):

    if not text_prompts:
        return [], [], []

    resized_frame = cv2.resize(frame, (640, 480))
    image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    with lock:
        current_prompts = text_prompts

    inputs = processor(text=current_prompts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.0,
        target_sizes=target_sizes
    )
    result = results[0]

    scores = result["scores"]
    keep = scores >= confidence_threshold
    boxes = result["boxes"][keep]
    scores = scores[keep]
    labels = result["labels"][keep]

    return boxes, scores, labels

# Function to draw bounding boxes and labels on the frame
def draw_detections(frame, boxes, scores, labels):

    for box, score, label in zip(boxes, scores, labels):
        box = box.cpu().numpy().astype(int)
        score = score.cpu().item()
        label_idx = label.cpu().item()

        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{text_prompts[label_idx]}: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main function to run the object detection pipeline
def main():

    parser = argparse.ArgumentParser(description='Zero-shot object detection with OWL-ViT')
    parser.add_argument('--input', type=str, default=None, help='Path to video file or webcam index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--log', type=str, default='detections.csv', help='Log file path (default: detections.csv)')
    args = parser.parse_args()

    global text_prompts
    text_prompts = [
        "a lightbulb",
        "a matchstick",
        "a monitor",
        "a lion",
        "a gaming console"
    ]

    root = tk.Tk()
    dashboard = SimpleDashboard(root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()

    if device == "cuda":
        model = model.half()
        print("Using half-precision floating point (FP16)")

    log_file = setup_logger(args.log)
    input_source = args.input if args.input else 0
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print("Error opening video source")
        return

    prompt_thread = threading.Thread(target=update_prompts, daemon=True)
    prompt_thread.start()

    frame_number = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, scores, labels = process_frame(frame, processor, model, device, args.confidence)
            current_detections = [(text_prompts[label.cpu().item()], score.cpu().item()) for label, score in zip(labels, scores)]
            dashboard.update_detections(current_detections)
            frame = draw_detections(frame, boxes, scores, labels)
            for box, score, label in zip(boxes, scores, labels):
                log_detection(log_file, frame_number, text_prompts[label.cpu().item()], score.cpu().item(), box.cpu().numpy())
            cv2.imshow('Zero-Shot Detection', frame)
            root.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()

if __name__ == "__main__":
    main()
