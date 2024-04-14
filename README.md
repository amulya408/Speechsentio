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
- Overview
- Setup Instructions
- Table of Contents
- Folder Structure
- Models
- Datasets
- Approach Used
- Training & Evaluation
- How to Render Code
- Features and Benefits
- Results
- Authors
- References
- License

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
Code can be rendered and executed using Jupyter notebooks available in the notebooks directory. Alternatively, Python scripts in the src folder can be run directly.

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

