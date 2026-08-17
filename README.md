# Multi-Label Retrieval of Image Using Deep Co-Image-Label Hashing

## Project Overview

This project predicts image labels using the Deep Co-Image-Label Hashing (DCILH) approach.

The application takes an input image, generates an image hash using DCT-based hashing, processes the hash using tokenization, and uses a trained deep learning model to predict the corresponding image label.

## Features

- Upload an image
- Generate image hash
- Process image using tokenization
- Predict image labels using a trained deep learning model
- Display predicted labels
- Streamlit web interface
- Live deployed application

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Scikit-learn

## How It Works

Image Upload  
↓  
Image Hash Generation  
↓  
Tokenization  
↓  
Deep Learning Model  
↓  
Label Prediction  
↓  
Predicted Label

## Model

The project uses a trained TensorFlow/Keras model with:

- Embedding layer
- LSTM encoder
- RepeatVector
- LSTM decoder
- TimeDistributed Dense layer
- Softmax activation

## Live Demo

[Streamlit App](https://multi-label-retrieval-of-image-using-deep-co-image-label-hashi.streamlit.app/)

## Example

**Input:** Image of a dog jumping over a log

**Output:**  
`a black dog leaps over a log`

## Project Structure

```text
DeepImageLabel/
│
├── app.py
├── DeepImageHashing.py
├── create_tokenizers.py
├── requirements.txt
├── Dataset.zip
│
└── model/
    ├── model.json
    ├── model_weights.h5
    ├── image_tokenizer.pkl
    ├── label_tokenizer.pkl
    ├── captions.txt
    ├── X.txt.npy
    └── Y.txt.npy