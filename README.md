# [Data_Project_Master -> Main AI Project About Heart Failure Prediction - Kalp Yetmezliği Risk Tahmini İçeren Yapay Zeka Projesi]

## 📌 Overview
Provide a 2-3 sentence overview of what this project does and the exact engineering challenge it solves. 
*Example:* This repository features an end-to-end Computer Vision pipeline designed to detect and track objects in real-time. By implementing model optimization techniques like TensorRT conversion and dataset stratification, inference latency was significantly reduced while maintaining high tracking precision.

## 🚀 Key Features
- **Custom Model Training:** Fine-tuned YOLOv5/EfficientNet architectures using Transfer Learning.
- **Model Optimization:** Converted PyTorch weights (`.pt`) to ONNX and TensorRT formats for edge deployment.
- **Data Engineering:** Automated image preprocessing and data stratification utilizing OpenCV.
- **Robust Backend:** Integrated with a Django REST API backed by a optimized PostgreSQL database.

## 🛠️ Tech Stack & Architecture
- **AI / Computer Vision:** PyTorch, OpenCV, YOLOv5, TensorRT, Scikit-learn
- **Backend & Database:** Python, Django REST Framework, PostgreSQL
- **DevOps / Infrastructure:** Linux/Ubuntu, Git, Docker

## 📊 Performance Metrics & Results
*Recruiters and engineers love numbers. Highlight the quantifiable results of your project here.*
- **Latency Reduction:** Decreased inference latency by **40%** (from 45ms to 27ms) using TensorRT optimization.
- **Accuracy Improvement:** Achieved a **+5.2% increase in mAP@0.5** via advanced data augmentation and dataset engineering.
- **Database Speed:** Reduced query response times by **150ms** through strategic database replication setups and indexing.

## 💻 Installation & Setup

Follow these steps to run the project locally:

```bash
# Clone the repository
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
