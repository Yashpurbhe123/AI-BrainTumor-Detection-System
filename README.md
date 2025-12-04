# 🧠 AI Brain Tumor Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)

## 📌 Project Overview

The **AI Brain Tumor Detection System** is a deep learning-based application designed to classify MRI brain scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

Leveraging the power of **Transfer Learning** with the **VGG16** architecture, this system achieves high accuracy in detecting brain tumors. The application is deployed using **Streamlit**, providing a user-friendly interface for medical professionals and students to upload MRI images and get instant predictions with probability confidence scores.

## 🚀 Features

-   **Upload MRI Images:** Support for JPG, JPEG, and PNG formats.
-   **Advanced Preprocessing:** Automatic resizing and normalization of images for optimal model performance.
-   **Fast Prediction:** Real-time classification using a pre-trained VGG16 model.
-   **Confidence Visualization:** Interactive bar charts displaying the probability for each class.
-   **Downloadable Results:** Option to download the prediction results as a JSON file.
-   **Responsive UI:** Clean and intuitive interface built with Streamlit.

## 🛠️ Tech Stack

-   **Deep Learning:** TensorFlow, Keras (VGG16)
-   **Web Framework:** Streamlit
-   **Image Processing:** OpenCV, Pillow (PIL), NumPy
-   **Visualization:** Plotly, Matplotlib, Seaborn
-   **Language:** Python

## 📂 Dataset

The model is trained on a comprehensive dataset of MRI brain scans, categorized into four classes:

1.  **Glioma Tumor**
2.  **Meningioma Tumor**
3.  **Pituitary Tumor**
4.  **No Tumor**

The dataset includes thousands of augmented images to ensure robust model performance.

## 🏗️ Model Architecture

The system utilizes **VGG16**, a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford.
-   **Base Model:** VGG16 (pre-trained on ImageNet)
-   **Custom Layers:** Flatten, Dense, Dropout, and Output layers added for specific tumor classification.
-   **Optimization:** Adam optimizer with categorical cross-entropy loss.

## 💻 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yashpurbhe123/AI-BrainTumor-Detection-System.git

    cd AI-BrainTumor-Detection-System
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the app:**
    -   Open your browser (usually at `http://localhost:8501`).
    -   Upload an MRI image using the file uploader.
    -   Click **"Run Prediction"**.
    -   View the predicted class and confidence score.



## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis or advice.
