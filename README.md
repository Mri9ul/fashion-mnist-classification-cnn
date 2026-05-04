# 👕 Fashion MNIST Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify Fashion MNIST images into 10 clothing categories.

---

## 📌 Objective

The objective of this project is to build a deep learning model that can classify grayscale fashion images such as shirts, shoes, bags, trousers, and other clothing items.

---

## 📊 Dataset

The project uses the **Fashion MNIST** dataset.

Fashion MNIST contains grayscale images of fashion products.

* **Training images:** 60,000
* **Test images:** 10,000
* **Image size:** 28 × 28 pixels
* **Number of classes:** 10

---

## 🏷️ Classes

The model predicts one of the following classes:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

---

## ⚙️ Project Workflow

1. Dataset loading
2. Data visualization
3. Image normalization
4. Image reshaping for CNN
5. CNN model building
6. Model training
7. Model evaluation
8. Classification report
9. Confusion matrix
10. Single image prediction
11. Model saving
12. Streamlit app development

---

## 🧠 Model Architecture

The CNN model contains:

* Convolutional layers
* MaxPooling layers
* Flatten layer
* Dense hidden layer
* Output layer for 10 classes

---

## 🖥️ Streamlit App

A Streamlit web app is included for interactive prediction.

### Features

* Upload image
* Convert image to grayscale
* Resize image to 28 × 28
* Predict clothing category
* Show confidence score
* Display class probability chart

Run the app:

```bash
streamlit run app.py
```

---


Give it a star ⭐ on GitHub!
