import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import wave

# Set the FFT size
n_fft = 8000

# Set the hop length
hop_length = 50

# Set the number of Mel bands
n_mels = 128

# Set the path to the directory containing the wave files
wav_dir = 'model_creation/val_data/'

# Set the path to the directory where the spectrograms will be saved
spec_dir = 'model_creation/val_spec_dir/'

# Loop over each class of wave files
for class_name in os.listdir(wav_dir):
    class_path = os.path.join(wav_dir, class_name)
    
    # Create a new directory for the spectrograms of this class
    new_spec_dir = os.path.join(spec_dir, class_name)
    os.makedirs(new_spec_dir, exist_ok=True)
    
    # Loop over each wave file in the class
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        # Compute the spectrogram
        signal, samplerate = librosa.load(file_path, sr=None, duration=1, mono=True)
        spectrogram = librosa.stft(signal,hop_length=hop_length)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        
        # Save the spectrogram as an image file
        spec_path = os.path.join(new_spec_dir, file_name[:-4] + '.jpg')
        plt.imsave(spec_path, spectrogram_db, cmap='gray')
