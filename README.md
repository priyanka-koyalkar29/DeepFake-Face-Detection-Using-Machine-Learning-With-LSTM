# DeepFake-Face-Detection-Using-Machine-Learning-With-LSTM
A machine learning project for detecting deepfake videos using Xception + LSTM.
# DeepFake Face Detection using CNN + LSTM 🎭📹

A hybrid deep learning-based project to detect DeepFake videos using **Xception CNN** for feature extraction and **LSTM** for temporal analysis. Built as part of my final year academic project, this model identifies manipulated videos based on facial features and time-sequence patterns.

---

## 📌 Table of Contents

- [🔍 Problem Statement](#-problem-statement)
- [🧠 Approach](#-approach)
- [🛠️ Technologies Used](#️-technologies-used)
- [📁 Dataset Used](#-dataset-used)
- [🚀 How to Run the Project](#-how-to-run-the-project)
- [📊 Results](#-results)
- [📌 Future Work](#-future-work)

---

## 🔍 Problem Statement

With the rise of AI-generated videos (DeepFakes), there's an increasing risk of misinformation, identity theft, and political manipulation. This project aims to build a robust detection system using deep learning techniques that can analyze video frames and detect whether a video is real or fake.

---

## 🧠 Approach

1. **Face Detection** using MTCNN
2. **Feature Extraction** from each frame using Xception (pre-trained on ImageNet)
3. **Temporal Pattern Learning** using LSTM to capture frame sequence behavior
4. **Binary Classification**: Real (0) or Fake (1)

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas / Matplotlib
- MTCNN (Multi-task Cascaded CNN)
- Flask (for web deployment)

---

## 📁 Dataset Used

- **FaceForensics++** (A benchmark DeepFake dataset with real and fake videos)
> Note: Due to storage limitations, datasets are not included in this repo. You can download them from [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/DeepFake-Face-Detection-LSTM.git
cd DeepFake-Face-Detection-LSTM
