import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fashion_mnist_cnn_model.keras")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.set_page_config(
    page_title="Fashion MNIST Classification",
    page_icon="👕"
)

st.title("👕 Fashion MNIST Classification using CNN")

st.write(
    "Upload a clothing image and the model will classify it into one of the Fashion MNIST categories."
)

st.info(
    "This model works best with simple grayscale clothing images similar to the Fashion MNIST dataset."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", use_container_width=True)

    # Resize to Fashion MNIST size
    image_resized = image.resize((28, 28))

    # Invert image if background is mostly white
    image_array = np.array(image_resized)
    if image_array.mean() > 127:
        image_resized = ImageOps.invert(image_resized)
        image_array = np.array(image_resized)

    # Normalize and reshape
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction[0])

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence Score: {confidence:.2f}")

    st.subheader("Class Probabilities")

    probabilities = {
        class_names[i]: float(prediction[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(probabilities)