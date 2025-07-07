# 📑 Signature Detection using CNNs


## 🗂️ Project Overview

This project presents the development of a **Convolutional Neural Network (CNN)** model to classify and identify handwritten signatures belonging to different individuals. The dataset consists of **16 images**, each containing **12 rows** with **4 signatures per row**. *Each row in every image represents a seperate individual to whom the signatures belong to.*

The goal is to segment these signatures, preprocess them, and train a CNN to recognize which signature belongs to which person. Labels follow the format **`imagenumber_rownumber`**, indicating the image and row the signature came from.  

The project also compares CNN-based feature extraction with manual techniques like **HOG** and **SIFT**, evaluating model performance using **precision**, **recall**, **F1 score**, **accuracy**, and training/testing loss curves.

---

## 🔧 Tools and Technologies Used

**Programming Language:** Python

**Libraries:**
- OpenCV
- NumPy
- Keras
- Matplotlib
- scikit-image

---

## 🧩 Techniques Used

- **Preprocessing:** Grayscale conversion, binary thresholding, contour extraction, resizing to 128×128 px  
- **Labeling:** Labels formatted as `imagenumber_rownumber`  
- **CNN Model:** Convolutional, pooling, dropout, dense, softmax layers  
- **Manual Feature Extraction:** HOG/SIFT  
- **Evaluation:** Accuracy, precision, recall, F1 score, loss/accuracy plots

---

## ⚙️ Data Loader Function

The data loader:
- Loads images and labels from a structured directory.
- Reads each signature in grayscale and resizes to 128×128 px.
- Labels each image as `imagenumber_rownumber`.
- Returns NumPy arrays for training/testing.

---

## 🚀 Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Signature-Detection-CNN.git
   cd Signature-Detection-CNN

2. **Open run the Jupyternotebook**
3. **Pip install all the libraries in the notebook**
4. **Set the input and output data directories after dowloading the dataset**
5. **Train the model**
6. **Run on the test set**
