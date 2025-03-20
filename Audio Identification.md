!pip install librosa==0.9.2 soundfile==0.10.3.post1

import librosa
import soundfile as sf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# 1. Load audio files and extract features
def extract_features(file_name):
    """
    Extracts features (MFCCs) from an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs_processed

# Set the path to your audio dataset
audio_dataset_path = '/content/drive/MyDrive/Bird Audio/Bird Audio'  # Replace with the actual path

features = []
labels = []

# Iterate through audio files and extract features
for folder in os.listdir(audio_dataset_path):
    folder_path = os.path.join(audio_dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.wav'):  # Adjust file extension if needed
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(folder)  # Assuming folder names represent bird species

# 2. Prepare data for training
# Convert features
!pip install librosa==0.9.2 soundfile==0.10.3.post1 pydub

import librosa
import soundfile as sf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment


# 1. Define a function to extract features from audio files
def extract_features(file_name):
    try:
        # Convert mp3 to wav using pydub
        audio = AudioSegment.from_mp3(file_name)
        audio.export("temp.wav", format="wav")  # Export as temporary wav file

        # Load the temporary wav file using librosa
        y, sr = librosa.load("temp.wav", sr=None)

        # Remove the temporary wav file
        os.remove("temp.wav")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs_processed

# 2. Load audio files and extract features
audio_dataset_path = '/content/drive/MyDrive/Bird Audio/Bird Audio'  # Replace with your dataset path
features = []
labels = []

for folder in os.listdir(audio_dataset_path):
    folder_path = os.path.join(audio_dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.mp3'):  # Check for .mp3 files
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(folder)

# ... (Rest of the code remains the same, including model training and testing)

!pip install librosa==0.9.2 soundfile==0.10.3.post1 pydub

import librosa
import soundfile as sf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

# 1. Define a function to extract features from audio files
def extract_features(file_name):
    try:
        # Convert mp3 to wav using pydub
        audio = AudioSegment.from_mp3(file_name)
        audio.export("temp.wav", format="wav")  # Export as temporary wav file

        # Load the temporary wav file using librosa
        y, sr = librosa.load("temp.wav", sr=None)

        # Remove the temporary wav file
        os.remove("temp.wav")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs_processed

# 2. Load audio files and extract features
audio_dataset_path = '/content/drive/MyDrive/Bird Audio/Bird Audio'  # Replace with your dataset path
features = []
labels = []

for folder in os.listdir(audio_dataset_path):
    folder_path = os.path.join(audio_dataset_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.mp3'):  # Check for .mp3 files
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(folder)

# 3. Prepare data for training
features = np.array(features)
label_encoder = LabelEncoder()
labels = to_categorical(label_encoder.fit_transform(labels))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 4. Build and train the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(features.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# 5. Save the trained model (optional)
model.save('/content/drive/MyDrive/bird_sound_classifier.h5')  # Save to Google Drive

# --- Testing Part ---
# 1. Define a function to predict bird species
def predict_bird_species(audio_file_path):
    features = extract_features(audio_file_path)  # Extract features using the same function as before

    # Check if features extraction was successful
    if features is None:
        print(f"Error: Could not extract features from {audio_file_path}. Skipping prediction.")
        return None  # Or handle the error in a different way

    features = features.reshape(1, -1)  # Reshape for model input
    prediction = model.predict(features)  # Make prediction
    predicted_class_index = np.argmax(prediction)  # Get predicted class index
    predicted_bird_species = label_encoder.inverse_transform([predicted_class_index])[0]  # Decode to bird species label
    return predicted_bird_species

# 2. Load the trained model (if you saved it)
# model = load_model('bird_sound_classifier.h5') # Uncomment if you saved the model
# 3. Example usage for testing
new_audio_file = '/content/drive/MyDrive/Bird Audio/Bird Audio/Baudo Guan_sound/Baudo Guan10.mp3'  # Replace with your test audio file path
predicted_species = predict_bird_species(new_audio_file)

# Check if prediction was successful
if predicted_species is not None:
    print("Predicted bird species:", predicted_species)
