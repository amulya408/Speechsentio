# SpeechSentio: AI-powered Speech Therapy with Emotion Analysis

## Overview
This repository contains the implementation of SpeechSentio, an AI-powered speech therapy system that integrates emotion analysis with stuttering detection. The system aims to provide precise and personalized speech therapy interventions by leveraging advanced signal processing techniques and machine learning algorithms.

## Setup Instructions
To set up and run SpeechSentio locally, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Download the necessary datasets or use your own speech data.
4. Run the main script to train the models and evaluate the system's performance.

## Table of Contents

- [SpeechSentio: AI-powered Speech Therapy with Emotion Analysis](#speechsentio-ai-powered-speech-therapy-with-emotion-analysis)
  - [Overview](#overview)
  - [Setup Instructions](#setup-instructions)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Approach Used](#approach-used)
  - [Training & Evaluation](#training--evaluation)
  - [How to Render Code](#how-to-render-code)
  - [Features and Benefits](#features-and-benefits)
  - [Results](#results)
  - [Authors](#authors)
  - [References](#references)
  - [License](#license)


## Models
The models directory contains the following trained models:

1. **Word Repetition Model**: `word_repetition_model.h5`
   - Description: This model is trained to detect word repetition patterns in speech.
   - File: `word_repetition_model.h5`
   - Usage: Used for identifying instances of word repetition in speech data.

2. **Sound Repetition Model**: `sound_repetition_model.h5`
   - Description: This model is designed to recognize sound repetition occurrences in speech.
   - File: `sound_repetition_model.h5`
   - Usage: Employed for detecting repetitions of sounds in speech samples.

3. **Prolongation Model**: `prolongation_model.h5`
   - Description: This model is trained to identify instances of sound prolongation in speech.
   - File: `prolongation_model.h5`
   - Usage: Utilized for detecting prolongation patterns in speech data.

4. **Emotion Model**: `emotion_model.pkl`
   - Description: This model performs emotion recognition in speech data.
   - File: `emotion_model.pkl`
   - Usage: Used for predicting emotions expressed in speech samples.

These models are utilized in the SpeechSentio system for various speech therapy interventions and emotion analysis tasks.

## Datasets

### Stutter Detection Dataset (sep28k)
The `sep28k` dataset is used for stutter detection tasks in the SpeechSentio system.
- Description: This dataset contains speech samples annotated for stuttering instances.
- Source: [SEP-28k Dataset](https://www.kaggle.com/datasets/bschuss02/sep28k "SEP-28k Dataset")
- Usage: Used for training and evaluating stutter detection models in SpeechSentio.

### Emotion Classification Dataset (RAVDESS)
The `RAVDESS` dataset is utilized for emotion classification tasks in the SpeechSentio system.
- Description: This dataset consists of speech recordings portraying various emotions.
- Source: [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio "RAVDESS Dataset")
- Usage: Employed for training and evaluating emotion classification models in SpeechSentio.

These datasets are integral to the development and evaluation of the SpeechSentio system, providing essential labeled data for training machine learning models.


## Approach Used
SpeechSentio utilizes a dual-branch architecture for simultaneous emotion recognition and stutter detection. Emotion analysis is performed using Mel-frequency cepstral coefficients (MFCCs) and Multi-Layer Perceptron (MLP) classifiers, while stutter detection employs decision trees and K-Nearest Neighbors (KNN) algorithms trained on speech features.

## Training & Evaluation
Training and evaluation of the models are performed using the datasets available in the data directory. Performance metrics such as accuracy, precision, recall, sensitivity, and F1 score are calculated to assess the models' effectiveness.

## How to Render Code

### Data Preprocessing and Feature Extraction:

1. Preprocess the raw speech data by implementing noise reduction, segmentation, and feature extraction techniques.
2. Utilize algorithms like signal processing methods (e.g., Fast Fourier Transform) and feature extraction libraries (e.g., librosa) to extract relevant features such as MFCCs, pitch, energy, and formants.

### Model Training:

1. Design and train machine learning models for emotion detection and stutter detection using the preprocessed features.
2. Consider using algorithms like Multi-Layer Perceptron (MLP), Convolutional Neural Networks (CNNs), decision trees, and K-Nearest Neighbors (KNN) for model training.
3. Use TensorFlow to build and train deep learning models if neural networks are preferred.

### Model Evaluation:

1. Evaluate the trained models using metrics such as accuracy, precision, recall, sensitivity, and F1 score.
2. Split the dataset into training and testing sets to assess the generalization performance of the models.

### Model Serialization:

1. Save the trained models to H5 files using TensorFlow's model serialization functionality.
2. This step ensures that the trained models can be reused without the need for retraining every time.

### Web Application Setup:

1. Initialize a Flask web application to serve as the interface for the speech therapy system.
2. Install Flask using pip if not already installed (`pip install Flask`).
3. Create the necessary directory structure for your Flask application (e.g., `app.py`, `templates/`, `static/`).

### Integrating Models with Flask:

1. Load the trained models (stored as H5 files) into the Flask application.
2. Define routes in Flask to handle incoming requests for speech analysis.
3. Implement the necessary logic to preprocess input speech data and feed it into the loaded models for prediction.

### Frontend Development:

1. Design the user interface of the web application using HTML, CSS, and JavaScript.
2. Utilize libraries such as Bootstrap and jQuery for responsive design and interactivity.
3. Create forms or input fields for users to upload speech data or interact with the application.

### Local Deployment:

1. Run the Flask application locally on your machine.
2. Start the Flask development server by executing the main Python file (e.g., `python app.py`) from the command line.
3. Access the web application through a web browser by navigating to `http://localhost:5000` or the port specified in your Flask application.

### Testing and Debugging:

1. Test the functionality of the web application by uploading sample speech data.
2. Debug any issues encountered during testing by examining error messages and logs.
3. Ensure that the application behaves as expected and provides accurate analysis and feedback to users.


## Features and Benefits
- Real-time analysis for pinpointing stuttering challenges.
- Emotional intelligence integration for creating a supportive therapy environment.
- Personalized pronunciation practice through in-depth analysis of individual speech patterns.
- Automated stutter detection and emotion recognition for efficient speech therapy interventions.

## Results
The following table summarizes the accuracies obtained by various ML algorithms:

| Algorithm            | Training Accuracy | Testing Accuracy |
|----------------------|-------------------|------------------|
| Decision Tree        | 80.375%           | 79.5%            |
| K-Nearest Neighbors  | 82%               | 85%              |
| Multi-Layer Perceptron | 75.35%          | 78%              |

## Authors
- Amulya Behara
- Lohitha Vasamsetty
- Pavithra Jasthi
- Abhinay Kunapuli

## References
[List of references cited in the research paper]

## License
This project is licensed under the MIT License.

