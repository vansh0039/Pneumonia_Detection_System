from pathlib import Path

import tensorflow as tf

from utils import preprocess_image


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "final_model.keras"


model = tf.keras.models.load_model(MODEL_PATH)


CLASS_NAMES = [

    "Normal",

    "Pneumonia"

]

def predict(uploaded_file):

    image, image_array = preprocess_image(uploaded_file)

    probability = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        prediction = CLASS_NAMES[1]

        confidence = probability

    else:

        prediction = CLASS_NAMES[0]

        confidence = 1 - probability

    return {
        "image": image,
        "prediction": prediction,
        "confidence": confidence,
        "probability": probability,
        "normal_probability": 1 - probability,
        "pneumonia_probability": probability
    }

