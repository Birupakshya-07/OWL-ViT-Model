# Zero-Shot Object Detection Using OWL-ViT


How It Works:

-Uses OWL-ViT for zero-shot object detection on a live webcam feed or video file.

-Takes custom text prompts (non-COCO objects) and detects them in real-time.

-Displays bounding boxes, labels, and confidence scores using OpenCV.

-Logs detections in a CSV file and updates a Tkinter dashboard with detected objects.

Challenges Faced:

-Model sometimes misclassifies objects due to zero-shot limitations.

-Real-time performance optimization, especially on CPUs.

=Ensuring efficient prompt updates without lag.

Future Improvements:

-Use faster models for real-time processing.

-Implement object tracking for stability across frames.







