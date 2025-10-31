# Speech Emotion Recognition with using dataset RAVDESS

This project recognizes emotions from speech using a deep learning model trained on the RAVDESS dataset. It extracts sound features (MFCCs) from audio and predicts emotions like Happy, Sad, Angry, or Neutral.

# What the Code Does

1) Loads the RAVDESS audio dataset
2) Extracts MFCC features from each audio file
3)Trains a CNN model to recognize emotions

Uses Gradio to make a simple web app for testing audio

# Libraries Used

1) librosa – for audio feature extraction
2) numpy, pandas – for data handling
3) scikit-learn – for label encoding and data split
4) tensorflow / keras – for building the CNN model
5) gradio – for making the user interface

# How to Run (in Google Colab)

1) Install required libraries
!pip install librosa soundfile numpy pandas scikit-learn tensorflow gradio joblib

2) Mount your Google Drive (if dataset is there)
from google.colab import drive
drive.mount('/content/drive')

3) Set the dataset path
DATASET_PATH = "/content/drive/MyDrive/ravdess/"

4) Run the training code
It will train the CNN and save the model as best_ravdess_model.h5.

5) Run the Gradio app cell
You’ll get a link to upload or record audio and see the predicted emotion.

# Dataset

RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song
Download from Kaggle

# Future Enhancements

1) Add more emotion classes from RAVDESS (like calm, fear, surprise).
2) Improve model accuracy using Mel-spectrograms.
3) Deploy as a real-time web or mobile app.
