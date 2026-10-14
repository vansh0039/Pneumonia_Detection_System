# 🩺 Pneumonia Detection System

A deep learning based **Computer Vision** application that detects **Pneumonia** from Chest X-ray images using **Transfer Learning (MobileNetV2)**. The project includes complete data exploration, model training, evaluation, inference, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Pneumonia is one of the leading causes of death due to respiratory infections. Early diagnosis from chest X-ray images can significantly improve treatment outcomes.

This project leverages a pretrained **MobileNetV2** model to classify chest X-ray images into:

- ✅ Normal
- 🦠 Pneumonia

The model was trained using transfer learning and evaluated using multiple performance metrics.

---

## 🚀 Features

- Chest X-ray image classification
- Transfer Learning using MobileNetV2
- Data preprocessing & visualization
- Model evaluation with multiple metrics
- ROC Curve
- Misclassified image analysis
- Streamlit Web Application
- Confidence Score Prediction
- Ready for deployment

---

## 📂 Project Structure

```
Pneumonia_Detection_System/
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Inference_and_Deployment.ipynb
│
├── models/
│   └── final_model.keras
│
├── sample_images/
│
├── results/
│   ├── classification_report.txt
│   ├── metrics.csv
│   ├── metrics.json
│   ├── history.json
│   ├── roc_curve.png
│   └── misclassified_images.png
│
├── app.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

Dataset:

**Chest X-Ray Images (Pneumonia)**

Classes:

- Normal
- Pneumonia

Images were resized to **224 × 224** before training.

---

# 🧠 Model

Base Model

- MobileNetV2 (ImageNet Weights)

Architecture

- MobileNetV2
- GlobalAveragePooling2D
- BatchNormalization
- Dense (128)
- Dropout
- Dense (1, Sigmoid)

Loss Function

- Binary Crossentropy

Optimizer

- Adam

---

# 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **88.94%** |
| Precision | **87.24%** |
| Recall | **96.41%** |
| F1 Score | **91.60%** |
| ROC-AUC | **96.74%** |
| Average Precision | **97.90%** |

---

# 📉 Classification Report

| Class | Precision | Recall | F1-Score |
|--------|-----------|----------|-----------|
| Normal | 0.93 | 0.76 | 0.84 |
| Pneumonia | 0.87 | 0.96 | 0.92 |

---

# 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Confusion Matrix
- ROC Curve

---

# 💻 Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Streamlit

---

# ▶️ Run Locally

Clone the repository

```bash
git clone https://github.com/your_username/Pneumonia_Detection_System.git
```

Go to the project

```bash
cd Pneumonia_Detection_System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📷 Sample Prediction

Upload a Chest X-ray image and the application predicts:

- Predicted Class
- Confidence Score

---

# 📒 Notebooks

### 01_Data_Exploration.ipynb

- Dataset Exploration
- Class Distribution
- Sample Images
- Image Statistics

### 02_Model_Training.ipynb

- Data Loading
- Transfer Learning
- Training
- Evaluation
- Metrics
- Saving Model

### 03_Inference_and_Deployment.ipynb

- Load Model
- Image Prediction
- Batch Prediction
- Visualization
- Deployment Preparation

---

# 📌 Future Improvements

- Grad-CAM Visualization
- Model Explainability (XAI)
- Multi-class Pneumonia Detection
- Ensemble Learning
- Docker Deployment
- Hugging Face Deployment

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

It should **not** be used as a substitute for professional medical diagnosis.

---

# 👨‍💻 Author

**Vansh Putalu**

GitHub: https://github.com/vansh0039
