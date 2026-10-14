from PIL import Image
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)


def preprocess_image(uploaded_file):
    """
    Load and preprocess an uploaded Chest X-ray image.
    """

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    return image, image_array