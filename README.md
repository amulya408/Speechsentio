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

Preprocess the raw speech data by implementing noise reduction, segmentation, and feature extraction techniques.Utilize algorithms like signal processing methods (e.g., Fast Fourier Transform) and feature extraction libraries (e.g., librosa) to extract relevant features such as MFCCs, pitch, energy, and formants.

### Model Training:

Design and train machine learning models for emotion detection and stutter detection using the preprocessed features.Consider using algorithms like Multi-Layer Perceptron (MLP), Convolutional Neural Networks (CNNs), decision trees, and K-Nearest Neighbors (KNN) for model training.Use TensorFlow to build and train deep learning models if neural networks are preferred.

### Model Evaluation:

Evaluate the trained models using metrics such as accuracy, precision, recall, sensitivity, and F1 score.Split the dataset into training and testing sets to assess the generalization performance of the models.

### Model Serialization:

Save the trained models to H5 files using TensorFlow's model serialization functionality.This step ensures that the trained models can be reused without the need for retraining every time.

### Web Application Setup:

Initialize a Flask web application to serve as the interface for the speech therapy system.Install Flask using pip if not already installed (`pip install Flask`).Create the necessary directory structure for your Flask application (e.g., `app.py`, `templates/`, `static/`).

### Integrating Models with Flask:

Load the trained models (stored as H5 files) into the Flask application.Define routes in Flask to handle incoming requests for speech analysis.Implement the necessary logic to preprocess input speech data and feed it into the loaded models for prediction.

### Frontend Development:

Design the user interface of the web application using HTML, CSS, and JavaScript. Utilize libraries such as Bootstrap and jQuery for responsive design and interactivity.Create forms or input fields for users to upload speech data or interact with the application.

### Local Deployment:

Run the Flask application locally on your machine.Start the Flask development server by executing the main Python file (e.g., `python app.py`) from the command line.Access the web application through a web browser by navigating to `http://localhost:5000` or the port specified in your Flask application.

### Testing and Debugging:

Test the functionality of the web application by uploading sample speech data.Debug any issues encountered during testing by examining error messages and logs.Ensure that the application behaves as expected and provides accurate analysis and feedback to users.


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
- [1]. Sadeen Alharbi, Madina Hasan, Anthony J H Simons, Shelagh Brumfitt, and Phil Green, 2017, ‘Stuttering from Transcripts: A Comparison of HELM and CRF Approaches’, eprints.whiterose.ac.uk, Vol. 1, pp. 1-11, 2017.
- [2]. Noeth, E., Wittenberg, T., Decher, M., & Dietrich, S. (2000). Automatic stuttering recognition using hidden Markov models. In Proceedings of the International Conference on Spoken Language Processing (ICSLP) (pp. 753-756).
- [3]. Kourkounakis, T., Hajavi, A., & Etemad, A. (2020). Detecting Multiple Speech Disfluencies Using a Deep Residual Network with Bidirectional Long Short-Term Memory. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3347-3351). Institute of Electrical and Electronics Engineers (IEEE).
- [4]. Al-Qatab, B. A., & Mustafa, M. B. (2021). Classification of Dysarthric Speech According to the Severity of Impairment: An Analysis of Acoustic Features. IEEE Access, 9, 18183-18194. Doi:10.1109/ACCESS.2021.3053335
- [5]. Lalitha, S., Tripathi, S., & Gupta, D. (2019). Enhanced speech emotion detection using deep neural networks. International Journal ofSpeech Technology, 22(3), 497–510. Doi:10.1007/s10772-018-09572-8
- [6]. H. Aouani and Y. B. Ayed, “Speech Emotion Recognition with deep learning,” Procedia Computer Science, vol. 176, pp. 251-260, 2020.
- [7]. I, Husbaan. (2022). Speech Emotion Recognition System Using Machine Learning. International Journal of Research Publication and Reviews, 3(5), pp. 2869-2880.
- [8]. Bhatia, G., Saha, B., Khamkar, M., Chandwani, A., & Khot, R. (2021). Stutter Diagnosis and Therapy System Based on Deep Learning. International Journal of Research Publication and Reviews, 5(1), 1-8.
- [9]. Barda, S. (2019). Recognition of rate of stuttering in patients having speech disorders. International Journal of Research Publication and Reviews, 3(1), 1-6.
- [10]. Mahendran,M.,Visalakshi, S., & Balaji, S. (2021). Dysarthria detection using CNN and MFCC feature extraction. International Journal of Research Publication and Reviews, 2(2), 1-7.
- [11]. Mustaqeem and S. Kwon, “A CNN-Assisted enhanced audio signal processing for speech emotion recognition,” Sensors, vol. 20, no. 1, p.183, 2019.
- [12]. M. Ghai, S. Lal, S. Duggal, and S. Manik, “Emotion recognition on speech signals using machine learning,” Mar. 2017, doi: 10.1109/icbdaci.2017.8070805.
- [13]. Apeksha Aggarwal, Akshat Srivastava, Ajay Agarwal, Nidhi Chahal, Dilbag Singh, Abeer Ali Alnuaim, Aseel Alhadlaq and Heung-No Lee, “Two-Way feature extraction for speech emotion recognition using deep learning,” Sensors, vol. 22, no. 6, p. 2378, Mar. 2022,1-11.
- [14]. T. Kourkounakis, A. Hajavi, and A. Etemad, “FluentNet: End-to-End Detection of Stuttered Speech Disfluencies with Deep Learning,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2986–2999, Jan. 2021, 1-13.
- [15]. Manas Jain, Shruthi Narayan, Pratibha Balaji, Bharath KP, Abhijit Bhowmick, Karthik R, Rajesh Kumar Muthu, “Speech Emotion Recognition using Support Vector Machine,” arXiv:2002.07590 [cs, eess], Feb. 2020, doi: https://arxiv.org/abs/2002.07590, 1-6.
- [16]. Suwon Shon, Pablo Brusco, Jing Pan, Kyu J. Han, Shinji Watanabe “Leveraging Pre-trained Language Model for Speech Sentiment Analysis” INTERSPEECH 2021 30 August – 3 September
- [17]. Bagus Tris Atmaja, Akira Sasou, “Sentiment Analysis and Emotion Recognition from Speech Using Universal Speech Representations”, 24 August 2022.
- [18]. Shakeel, A. Sheikh, Md Sahidullah , Fabrice Hirsc, Slim Ouni, “StutterNet: Stuttering Detection Using Time Delay Neural Network” , Jan. 2021
- [19]. Colin Lea, Zifang Huang, Jaya Narain, Lauren Tooley, Dianna Yee, Tien Dung Tran, “From User Perceptions to Technical Improvement: Enabling People Who Stutter to Better Use Speech Recognition”, April 23–28, 2023
- [20]. Phani Bhushan S, Vani H Y, D K Shivkumar, “Stuttered Speech Recognition using Convolutional Neural Networks”, 2021
- [21]. Badshah, A., Lee, S., & Kim, J. (2017). Speech Emotion Recognition from Spectrograms with Deep Convolutional Neural Network. International Journal of Research in Engineering and Technology, 6(6), 239-244.
- [22]. Li, S., Deng, L., & Huang, J. T. (2013). Hybrid Deep Neural Network – Hidden Markov Model Based Speech Emotion Recognition. International Journal of Research Publication and Reviews, 6(3), 312- 317.
- [23]. Mahesha, P., & Vinod, D. S. (2016). Automatic Segmentation and Classification of Dysfluencies in Stuttering Speech. International Journal of Research Publication and Reviews, 3(4), 12-20.
- [24]. Mahesha, P., & Vinod, D. S. (2017). LP-Hilbert Transform Based MFCC for Effective Discrimination of Stuttering Dysfluencies. International Journal of Research Publication and Reviews, 4(3), 2564-2571.
- [25]. Narendra, N.P., & Alku, P. (2019). Dysarthric speech classification from coded telephone speech using glottal features. Speech Communication, 110, 47-55.
- [26]. Zhao, J., Mao, X., & Chen, L. (2018). Speech emotion recognition using deep 1D & 2D CNN LSTM networks. Biomedical Signal Processing and Control, 47, 312-323.
- [27]. Issa, D., Demirci, M. F., & Yazici, A. (2020). Speech emotion recognition with deep convolutional neural networks. Biomedical Signal Processing and Control, 59, 101894.
- [28]. Alif Bin Abdul Qayyum, Asiful Arefeen, & Celia Shahnaz. (2019). Convolutional Neural Network (CNN) Based Speech-Emotion Recognition. International Journal of Research Publication and Reviews, 6(4), 122-130.
- [29]. Afroz, F., & Koolagudi, S. G. (2019). Recognition and Classification of Pauses in Stuttered Speech using Acoustic Features. International Journal of Research Publication and Reviews, 6(2), 12-20.
- [30]. Bhushan, P., Shivkumar, D. K., Vani, H. Y., & Sreeraksha, M. R. (Year). Stuttered Speech Recognition using Convolutional Neural Networks. International Journal of Research Publication and Reviews, Volume 9(Issue 12), pp. 250-254.
- [31]. GitHub Repository:
[stutter-classification](https://github.com/mitul-garg/stutter-classification/tree/main) - Repository containing code for stutter classification.





