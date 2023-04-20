import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import wave
# Set the FFT size
n_fft = 128

# Set the hop length
hop_length = 32

classes = ["wav"]
for cls in classes:
    # Set the path to the directory containing the wave files
    wav_dir = f'/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/{cls}_data'

    # Set the path to the directory where the spectrograms will be saved
    spec_dir = f'/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/{cls}_spec_test'

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
            signal, samplerate = librosa.load(file_path, sr=8000, duration=165, mono=True)
            spectrogram = librosa.stft(signal,hop_length=hop_length, n_fft=n_fft)
            spectrogram_abs = (abs(spectrogram))
            spec_db = librosa.amplitude_to_db(spectrogram_abs, ref=np.max)
            
            # Save the spectrogram as an image file
            spec_path = os.path.join(new_spec_dir, file_name[:-4] + '_log.jpg')
            # Set the figure size
            #plt.figure(figsize=(10, 4))

            # Plot the spectrogram
            librosa.display.specshow(spec_db, x_axis='time', y_axis='log', hop_length=hop_length)
            plt.imsave(spec_path, spec_db, cmap='gray')
