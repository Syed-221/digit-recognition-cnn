
# 🧠 Handwritten Digit Recognition using CNN & OpenCV

This project demonstrates a deep learning approach to recognizing handwritten digits using a Convolutional Neural Network (CNN). It leverages the popular MNIST dataset and integrates TensorFlow, OpenCV, and Gradio for training, prediction, and deployment.

---

## 📌 Overview

- 🔢 **Dataset**: MNIST (60,000 training, 10,000 testing images)
- 🧠 **Model**: Convolutional Neural Network (CNN)
- 🛠️ **Tech Stack**: Python, TensorFlow, OpenCV, Gradio, Google Colab
- 🌐 **UI**: Gradio Web Interface (Optional)

---

## 🗂️ How It Works

### 1. Load and Preprocess Data
- Images reshaped from (28, 28) to (28, 28, 1)
- Normalized pixel values to [0, 1]
- Labels converted to one-hot encoding

### 2. Build the CNN
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
```
### 3. Train the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```
### 4. Predict and Visualize
```python
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
print("Prediction:", np.argmax(model.predict(x_test[0].reshape(1,28,28,1))))
```
##🧪 Optional Enhancements

###🔍 OpenCV Integration

Resize and preprocess images manually

Use OpenCV to draw and capture real-time digits

###🖼️ Gradio App Interface
```python
def predict_digit(image):
    img = cv2.resize(image, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 28, 28, 1) / 255.0
    pred = model.predict(img).argmax()
    return f"Predicted Digit: {pred}"

gr.Interface(fn=predict_digit, inputs="image", outputs="text").launch()
```
### ✅ Results
Achieved over 98% accuracy on test dataset

Successfully deployed real-time digit recognizer using Gradio

### 📁 Files Included

digit_cnn_model.h5: Saved trained model

mnist_digit_classifier.ipynb: Full code in Jupyter/Colab format

app.py: Optional Gradio app

## 🧠 Author
**Syed Musharaf Hossain**  
[LinkedIn](https://www.linkedin.com/in/syed-musharaf-hossain) | [GitHub](https://github.com/Syed-221)
