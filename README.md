# Zero-Shot Object Detection Using OWL-ViT


This project implements real-time object detection using a zero-shot vision model, OWL-ViT, to identify objects that are not part of the standard COCO dataset. The system takes input from a webcam or a video file and processes each frame using OpenCV. It uses predefined custom object categories as text prompts, which are processed through OWL-ViT to detect and classify objects. The detections are displayed with bounding boxes, labels, and confidence scores on a live video feed. Additionally, detections are logged into a CSV file for further analysis, and a Tkinter-based dashboard updates in real-time to list detected objects.

One of the main challenges faced was ensuring the model generalizes well to unseen categories, as zero-shot learning can sometimes produce false positives or struggle with objects in complex scenes. Another challenge was optimizing performance for real-time inference, especially on CPUs, since OWL-ViT is a large model and benefits significantly from GPU acceleration. Future improvements could include integrating more efficient models for speed optimization, adding a tracking module to improve detection consistency across frames, and enhancing the user interface with interactive prompt updates. Deploying this system as a web application with Streamlit or FastAPI could also make it more accessible.








