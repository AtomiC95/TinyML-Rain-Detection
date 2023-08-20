import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

def plot_wave(file_path, subplot_position):
    spf = wave.open(file_path, "r")
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)

    # If Stereo
    if spf.getnchannels() == 2:
        print(f"{file_path} is stereo. Only mono files supported.")
        return

    # Create a time array
    num_frames = len(signal)
    frame_rate = spf.getframerate()
    time_array = np.linspace(0, num_frames / frame_rate, num_frames)

    samples_per_minute = 120 * frame_rate
    signal_two_minute = signal[:samples_per_minute]
    time_array_two_minute = time_array[:samples_per_minute]


    plt.subplot(2, 2, subplot_position) 
    plt.plot(time_array_two_minute, signal_two_minute)
    plt.title(file_path.split('/')[-1]) 
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

# List of your WAV files
files = [
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/heavy_rain.wav",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/light_rain.wav",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/medium_rain.wav",
    "/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/no_rain.wav"
]

plt.figure(figsize=(10, 8))

for idx, file in enumerate(files, 1):
    plot_wave(file, idx)

plt.tight_layout()
plt.show()
