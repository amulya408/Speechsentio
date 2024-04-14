from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import soundfile as sf
import os
from markupsafe import Markup
import pickle
import random
import time

app = Flask(__name__)

# Load your stutter classification models
word_repetition_model = load_model(r"C:\Users\BEHARA AMULYA\Downloads\Major project\stutter-classification-main\prolongation_model.h5")
sound_repetition_model = load_model(r"C:\Users\BEHARA AMULYA\Downloads\Major project\stutter-classification-main\sound_repetition_model.h5")
prolongation_model = load_model(r"C:\Users\BEHARA AMULYA\Downloads\Major project\stutter-classification-main\prolongation_model.h5")

# Load the emotion recognition model
emotion_model = pickle.load(open(r"C:\Users\BEHARA AMULYA\Downloads\Major project\emotion_model.pkl", "rb"))

# Emotion mapping
emotion_mapping = {'0': 'Neutral', '1': 'Calm', '2': 'Happy', '3': 'Sad', '4': 'Angry', '5': 'Fearful', '6': 'Disgust', '7': 'Surprised'}

# Function to extract features from audio file
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
    return mfccs.reshape(1, -1)

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        pad_size = 180 - len(result)
        if pad_size > 0:
            result = np.pad(result, (0, pad_size), 'constant')
    return result

# Function to classify stutter type
def classify_stutter(file_path, model):
    features = extract_features(file_path)
    prediction = model.predict(features)
    return prediction[0]

# Function to predict emotion
def predict_emotion(file_path):
    feature = extract_feature(file_path)
    feature = feature.reshape(1, -1)
    emotion_prediction = emotion_model.predict(feature)
    return emotion_prediction[0]

# Function to generate a random paragraph
def generate_random_paragraph():
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "All that glitters is not gold.",
        "In the midst of winter, I found there was, within me, an invincible summer.",
        "To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.",
        "Life is what happens when you're busy making other plans.",
        "The only limit to our realization of tomorrow will be our doubts of today.",
        "Success is not final, failure is not fatal: It is the courage to continue that counts.",
        "The greatest glory in living lies not in never falling, but in rising every time we fall.",
        "The purpose of our lives is to be happy."
    ]
    num_sentences = random.randint(3, 6)
    paragraph = " ".join(random.sample(sentences, num_sentences))
    return paragraph

def get_random_tongue_twister():
    tongue_twisters = [
        "How much wood would a woodchuck chuck\nIf a woodchuck could chuck wood?\nHe would chuck as much wood as a woodchuck would\nIf a woodchuck could chuck wood.",
        "I saw Susie sitting in a shoeshine shop.\nWhere she sits she shines, and where she shines she sits.",
        "She sells seashells by the seashore,\nThe shells she sells are surely seashells.\nSo if she sells shells on the seashore,\nI'm sure she sells seashore shells.",
        "Betty Botter bought some butter,\nBut she said the butter's bitter.\nIf I put it in my batter,\nIt will make my batter bitter.\nSo she bought some better butter,\nBetter than the bitter butter.\nAnd she put it in her batter,\nAnd her batter was not bitter.\nSo 'twas better Betty Botter\nBought some better butter.",
        "A proper copper coffee pot.\nI'm not a pheasant plucker,\nI'm a pheasant plucker's son.\nI'm only plucking pheasants\nTill the pheasant plucker comes.",
        "Six slippery snails slid silently seaward.\nSilly Sally swiftly shooed seven silly sheep.\nThe seven silly sheep Silly Sally shooed\nShilly-shallied south.\nThese sheep shouldn't sleep in a shack;\nSheep should sleep in a shed.",
        "I saw Susie sitting in a shoeshine shop.\nWhere she sits she shines, and where she shines she sits.",
        "How can a clam cram in a clean cream can?\nIf you must cross a course, cross cow across a crowded cow crossing,\nCross the cross coarse cow across the crowded cow crossing carefully.",
        "Fuzzy Wuzzy was a bear.\nFuzzy Wuzzy had no hair.\nFuzzy Wuzzy wasn't very fuzzy, was he?",
        "Near an ear, a nearer ear, a nearly eerie ear.\nA proper cup of coffee in a copper coffee cup."
    ]
    return random.choice(tongue_twisters)

def interpret_predictions(word_repetition_output, sound_repetition_output, prolongation_output, emotion_output):
    audio_analysis = "Audio Analysis:<br>"
    recommendations = "Recommendation:<br>"

    audio_analysis += "Word Repetition Prediction: "
    if word_repetition_output > 0.05:
        audio_analysis += "Word Repetition found"
        recommendations += "You are repeating the same word again and again. Practice slow and deliberate speech. Take a pause between words to allow for smoother communication.<br>"
    else:
        audio_analysis += "No Word Repetition"
    audio_analysis += "<br><br>"

    audio_analysis += "Sound Repetition Prediction: "
    if sound_repetition_output > 0.01:
        audio_analysis += "Sound Repetition found"
        recommendations += "Sound Repetition is found in your voice. Work on relaxation techniques to reduce tension. Focus on breathing exercises to promote a more natural speech flow.<br>"
    else:
        audio_analysis += "No Sound Repetition"
    audio_analysis += "<br><br>"

    audio_analysis += "Prolongation Prediction: "
    if prolongation_output > 0.05:
        audio_analysis += "Prolongation found"
        recommendations += "And Prolongation is evident in your voice. Practice controlled breathing to ease into speech sounds. Gradually increase the speed of speech while maintaining control.<br>"
    else:
        audio_analysis += "No Prolongation"
    audio_analysis += "<br><br>"

    audio_analysis += f"Emotion Prediction: {emotion_output}"
    audio_analysis += "<br><br>"

    # Check if any stutter type is detected, if not, print random paragraphs and tongue twisters
    if word_repetition_output > 0.05 or sound_repetition_output > 0.05 or prolongation_output > 0.05:
        recommendations += f"<br>Practice tongue twisters like these regularly to improve your speech:<br>"

        for i in range(3):
            recommendations += f"{i + 1}. {get_random_tongue_twister()}<br>"

        recommendations += "<br>Improve your fluency by reading these sentences slowly and clearly:<br>"

        for i in range(3):
            recommendations += f"{i + 1}. {generate_random_paragraph()}<br>"

    return Markup(audio_analysis + recommendations)


@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audioFile' not in request.files:
        return render_template('index3.html', message='No file part')

    file = request.files['audioFile']

    if file.filename == '':
        return render_template('index3.html', message='No selected file')

    if file:
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        word_repetition_output = classify_stutter(file_path, word_repetition_model)
        sound_repetition_output = classify_stutter(file_path, sound_repetition_model)
        prolongation_output = classify_stutter(file_path, prolongation_model)
        emotion_output = predict_emotion(file_path)

        result_message = interpret_predictions(word_repetition_output, sound_repetition_output, prolongation_output, emotion_output)

        return render_template('result.html', message=result_message, prediction=emotion_output)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
