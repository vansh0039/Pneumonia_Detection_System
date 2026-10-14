import streamlit as st
from predict import predict

# Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# Sidebar - Model Information
st.sidebar.title("📋 Model Information")
st.sidebar.markdown("""
### Architecture
- MobileNetV2

### Input Size
- 224 × 224

### Classes
- Normal
- Pneumonia

### Performance
- Accuracy : 88.94%
- Precision : 87.24%
- Recall : 96.41%
- F1 Score : 91.60%
- ROC-AUC : 96.74%
""")

# Main Page Header
st.title("🩺 Chest X-ray Pneumonia Detection")
st.caption("Deep Learning using MobileNetV2")

st.write("Upload a Chest X-ray image to predict whether it shows signs of pneumonia.")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a Chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Run prediction from predict.py
    result = predict(uploaded_file)

    # Display Image
    st.image(
        result["image"],
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prediction Display with color coding
    st.subheader("Prediction")
    if result["prediction"] == "Normal":
        st.success("🟢 Normal")
    else:
        st.error("🔴 Pneumonia")

    # Confidence Metric
    st.metric(
        "Confidence",
        f"{result['confidence'] * 100:.2f}%"
    )

    # Confidence Level Evaluation
    confidence = result["confidence"]
    if confidence >= 0.95:
        st.success("High Confidence Prediction")
    elif confidence >= 0.80:
        st.warning("Moderate Confidence Prediction")
    else:
        st.error("Low Confidence Prediction")

    # Class Probabilities Breakdown
    st.write("### Class Probabilities")
    st.write(f"🟢 Normal : {result['normal_probability'] * 100:.2f}%")
    st.write(f"🔴 Pneumonia : {result['pneumonia_probability'] * 100:.2f}%")

    # Medical Disclaimer
    st.info(
        "This model is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis."
    )

# About Section Footer
st.markdown("---")
st.subheader("About")
st.write("""
This application uses a **MobileNetV2** deep learning model trained on Chest X-ray images to classify images as:

- 🟢 Normal
- 🔴 Pneumonia
""")