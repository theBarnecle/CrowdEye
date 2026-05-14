# CrowdEye
👁️ CrowdEye
Real-Time Object Detection using YOLOv8

CrowdEye is a lightweight yet powerful real-time object detection system built using Ultralytics YOLOv8, OpenCV, and PyTorch.

It supports image, video, and live webcam inference with optional output saving and automatic GPU acceleration.

🚀 Features
Real-time detection (Webcam / Video / Image)
YOLOv8 model support (nano → custom weights)
Adjustable confidence threshold
Automatic CPU / CUDA selection
FPS counter for performance monitoring
Clean CLI interface
Optional output saving
🧠 How It Works

CrowdEye:

Loads a YOLOv8 model.
Accepts input from webcam, image, or video.
Performs inference frame-by-frame.
Draws bounding boxes with class labels & confidence scores.
Displays results in real time.
Optionally saves processed output.

Simple. Focused. No unnecessary abstraction.

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/crowdeye.git
cd crowdeye

(Replace with your actual GitHub URL. Obviously.)

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
pip install ultralytics opencv-python torch torchvision

Or use a requirements.txt file if provided.

▶ Usage
Webcam
python detect.py --source 0
Image
python detect.py --source image.jpg
Video
python detect.py --source video.mp4
Adjust Confidence
python detect.py --source video.mp4 --conf 0.4
Save Output
python detect.py --source video.mp4 --save

Custom directory:

python detect.py --source video.mp4 --save --save_dir results
⚙ Command-Line Arguments
Argument	Description	Default
--source	Image path, video path, or 0 for webcam	0
--model	YOLO model weights	yolov8n.pt
--conf	Confidence threshold	0.25
--save	Save processed output	Disabled
--save_dir	Output folder	output
🖥 Device Selection

CrowdEye automatically selects:

CUDA (GPU) if available
Otherwise defaults to CPU

No manual configuration required.

📂 Output Structure
crowdeye/
│
├── detect.py
├── output/
│   ├── result.jpg
│   └── result.mp4
📊 Performance Notes
FPS depends on:
Model size (yolov8n vs yolov8s etc.)
GPU availability
Input resolution
For higher FPS, use smaller models (e.g., yolov8n.pt).
🛠 Requirements
Python 3.8+
PyTorch
OpenCV
Ultralytics YOLOv8
