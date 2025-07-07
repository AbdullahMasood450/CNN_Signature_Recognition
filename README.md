# Signature Detection using CNNs

## 📑 Research Paper
[Report (PDF)](https://github.com/AbdullahMasood450/CNN_Signature_Recognition/blob/main/CNN_Signature_Recog.pdf)

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
- Loads images and forms a structured directory calssifying images as training, test and validation.
- Reads each signature in grayscale and resizes to 128×128 px.
- Labels each image as `imagenumber_rownumber`.
- Returns NumPy arrays for training/testing/validations.

---

## 🚀 Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Signature-Detection-CNN.git
   cd Signature-Detection-CNN
2. **Download the dataset**
3. **Open run the Jupyternotebook**
4. **Pip install all the libraries in the notebook**
5. **Set the input and output data directory link**
6. **Train the model**
7. **Run on the test set**

---

## 📊 Results

The **CNN Signature Recognition** model was trained and evaluated on the prepared dataset.  
Below are the summarized results:

- **Training Accuracy:** ~75.12%  
- **Validation/Test Accuracy:** ~9.94%  
- **Train Loss:** ~2.80  
- **Test Loss:** ~5.20  
- **Precision:** ~9%  
- **Recall:** ~12%  
- **F1 Score:** ~10%  

These results indicate that while the model performed well on the training data, its test accuracy and generalization remain low — highlighting overfitting and the need for further improvements like **data augmentation**, **hyperparameter tuning**, or exploring **more advanced architectures**.

---

## 📈 Visualizations

### Loss and Accuracy Plots
![Loss_Accuracy](https://github.com/user-attachments/assets/beaf0acd-3a20-4211-82fd-47d6e67a0cd0)

### Precisionn Recall and F1 confusion Matrix
![Precison, Recall, F1](https://github.com/user-attachments/assets/8d937a26-acbd-4b4b-ac97-fc70ccfc437d)


   
