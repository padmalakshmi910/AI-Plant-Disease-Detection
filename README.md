# AI-Based Plant Disease Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to identify plant leaf diseases from uploaded images.

The application predicts the disease category and displays prediction confidence, symptoms, causes, and prevention information.

## Technologies Used
- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Matplotlib
- Seaborn
- Google Colab

## Dataset
The project uses the PlantVillage dataset.

- Total images: 20,638
- Number of classes: 15
- Training images: 16,511
- Validation images: 4,127
- Image size: 128 x 128 pixels

## CNN Model
The CNN contains:
- Three convolutional layers
- Max pooling layers
- Flatten layer
- Dense layer
- Dropout layer
- Softmax output layer

The model was trained using the Adam optimizer and sparse categorical cross-entropy loss.

## GPU Acceleration
Model training was performed in Google Colab using an NVIDIA Tesla T4 GPU.

TensorFlow was used to detect the available GPU and perform model training.

GPU verification:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Results
- Validation accuracy: 91.54%
- Validation loss: 0.2527
- Number of predicted classes: 15

Training accuracy graphs and confusion matrix are included in the repository.

## Application Features
- Upload plant leaf images
- Predict plant disease using CNN
- Display prediction confidence
- Show top three predictions
- Display disease information
- Provide basic prevention suggestions

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The trained model is downloaded from Google Drive when the application starts. The model file must be accessible through its sharing link.

## Disclaimer
This project is developed for educational purposes. Predictions should not replace professional agricultural diagnosis.
