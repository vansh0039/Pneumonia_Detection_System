# 🩺 Pneumonia Detection System

A deep learning computer vision project that classifies chest X-ray images as **Normal** or **Pneumonia** using transfer learning with MobileNetV2. The project includes data exploration, model training, evaluation, inference, and a Streamlit web application.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose or make clinical treatment decisions.

## 📌 Project Overview

Pneumonia is a respiratory infection that can cause serious illness. This project explores how deep learning can be applied to chest X-ray image classification.

A pretrained MobileNetV2 model is adapted using transfer learning to perform binary classification:

- **Normal:** No pneumonia detected by the model.
- **Pneumonia:** The model predicts pneumonia.

The application provides a predicted class and confidence score. These are model outputs, not confirmed medical findings.

## ✨ Features

- Chest X-ray image classification
- Transfer learning using MobileNetV2
- Image preprocessing and data exploration
- Model training and performance evaluation
- Classification report and evaluation metrics
- ROC curve analysis
- Misclassification analysis
- Confidence score prediction
- Streamlit web interface
- Clinical decision-threshold analysis

## 🗂️ Project Structure

```text
Pneumonia_Detection_System/
├── models/
│   └── final_model.keras
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Inference_and_Deployment.ipynb
│   └── 04_Clinical_Decision_Threshold_Optimization.ipynb
├── results/
│   ├── classification_report.txt
│   ├── history.json
│   ├── metrics.csv
│   ├── metrics.json
│   └── roc_curve.png
├── app.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

The dataset contains two classes:

- Normal
- Pneumonia

Images are resized to **224 × 224 pixels** for model input.

The original dataset is not included in this repository. Refer to the dataset provider's terms before obtaining or redistributing images.

## 🧠 Model Architecture

**Base model:** MobileNetV2 with ImageNet pretrained weights.

The classification head consists of:

1. MobileNetV2 feature extraction
2. Global Average Pooling
3. Batch Normalization
4. Dense layer with 128 units
5. Dropout
6. Dense output layer with one sigmoid unit

**Training configuration**

- Loss function: Binary Crossentropy
- Optimizer: Adam
- Learning approach: Transfer learning

See the training notebook for the implementation and training details.

## 📈 Model Performance

The following results are reported for the evaluated model:

| Metric | Score |
|---|---:|
| Accuracy | 88.94% |
| Precision | 87.24% |
| Recall | 96.41% |
| F1 Score | 91.60% |
| ROC-AUC | 96.74% |
| Average Precision | 97.90% |

These metrics describe performance on the evaluated dataset and should not be interpreted as evidence of clinical effectiveness or generalization to other hospitals or patient populations.

See the `results/` directory for the saved evaluation artifacts.

## 📋 Classification Report

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Normal | 0.93 | 0.76 | 0.84 |
| Pneumonia | 0.87 | 0.96 | 0.92 |

The report suggests higher recall for the Pneumonia class than for the Normal class. This makes it important to evaluate false positives, false negatives, and the trade-offs associated with different decision thresholds.

## 🧪 Evaluation

The project includes analysis using:

- Accuracy, precision, recall, and F1 score
- ROC-AUC and Average Precision
- Classification report
- ROC curve
- Training history
- Misclassification analysis
- Decision-threshold optimization

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vansh0039/Pneumonia_Detection_System.git
cd Pneumonia_Detection_System
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The trained model is stored at `models/final_model.keras`. Ensure the application and prediction code use the correct model path.

## 📓 Notebooks

The notebooks document the project workflow:

- **01 — Data Exploration:** Dataset inspection, class distribution, and image analysis.
- **02 — Model Training:** Transfer learning, training, evaluation, and model saving.
- **03 — Inference and Deployment:** Model loading, predictions, and deployment preparation.
- **04 — Clinical Decision Threshold Optimization:** Threshold selection and evaluation of classification trade-offs.

## 🔭 Future Improvements

- Grad-CAM visual explanations
- Further model interpretability and error analysis
- External validation on independent datasets
- Robustness testing across different image sources
- Docker-based deployment
- Deployment on Hugging Face Spaces

## ⚠️ Medical Disclaimer

This project is intended strictly for educational and research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Predictions may be incorrect and must not be used to make clinical decisions.

## 👨‍💻 Author

**Vansh Putalu**

GitHub: [@vansh0039](https://github.com/vansh0039)

Project repository: [Pneumonia Detection System](https://github.com/vansh0039/Pneumonia_Detection_System)
