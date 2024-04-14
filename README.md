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
The models directory contains trained models for emotion detection and stutter detection.

## Datasets
The data directory includes both raw and processed datasets used for training and evaluation.

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

