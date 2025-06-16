
import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load model
model = load_model("digit_cnn_model.h5")

# Define prediction function
def predict_digit(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image).argmax()
    return f"Predicted Digit: {prediction}"

# Launch Gradio app
interface = gr.Interface(fn=predict_digit, inputs="image", outputs="text", live=True)
interface.launch()
