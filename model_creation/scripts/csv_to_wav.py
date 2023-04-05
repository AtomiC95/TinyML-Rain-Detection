import copy
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

from pathlib import Path

def load_csv_with_pandas():
    heavy_rain = "heavy_rain"
    light_rain = "light_rain"
    no_rain = "no_rain"
    medium_rain = "medium_rain"
    # assume we have columns 'time' and 'value'
    # df = pd.read_csv('model_creation/model_data/raw_data')

    # # compute sample rate, assuming times are in seconds
    # times = df['time'].values
    # n_measurements = len(times)
    # timespan_seconds = times[-1] - times[0]
    # sample_rate_hz = int(n_measurements / timespan_seconds)
    with open('model_creation/model_data/raw_data/HEAVY_RAIN_36mmperH.txt', 'r') as f:
        data_str = f.read()

    # Step 2: convert data to numpy array
    data_list = data_str.strip().split('\n')
    data = np.array([int(d) for d in data_list])
    # write data
    # i2s0_left = (df['i2s0_left'].values)
    # i2s0_right = (df['i2s0_right'].values)
    # i2s1_left = (df['i2s1_left'].values)
    # i2s1_right = (df['i2s1_right'].values)
    # data_neu = np.int16(data)
    # print(data_neu)
    # sf.write('recording.wav', data_neu, 44100)
    samplerate = 8000
    # t = np.linspace(0., 1., samplerate)
    # amplitude = np.iinfo(np.int16).max
    # data = amplitude * np.sin(2. * np.pi * fs * t)
    write("model_creation/model_data/wav_data/" + f{heavy_rain}, samplerate, data.astype(np.int16))
    # write("i2s0_right.wav", samplerate, i2s0_right.astype(np.int16))
    # write("i2s1_left.wav", samplerate, i2s1_left.astype(np.int16))
    # write("i2s1_right.wav", samplerate, i2s1_right.astype(np.int16))

def convert_out_to_np(path: Path) -> np.ndarray:
    """convert out.txt to array, only works with one microphone"""
    with open(path, "r") as f:
        start_found = False
        channels =([]) # ([], [], [], [])
        for line in f:
            if (start_found
                and len(line) > 0
                and len(line) < 25
                and "," in line): # add to array after start found
                for i, num in enumerate(line.split(",")):
                    channels[i].append(int(float(num)))
            if "i2s0_left" in line:
                start_found = True
    data = np.array(channels).T
    return data

def save_signal_to_wav(data: np.ndarray, samplerate: int, dtype: np.dtype, path: Path = Path("test.wav")) -> None:
    """save a singal to a .wav file"""
    data_copy = copy.deepcopy(data)
    # normalize such that max value in signal has max amplitude
    # the resulting file will sound louder
    norm_factor = np.iinfo(dtype).max // max(data)
    data_copy *= norm_factor
    write(path, samplerate, data_copy.astype(dtype))

def plot_signal_x_samples(data: np.ndarray) -> None:
    plt.figure()
    plt.plot(data)
    plt.xlabel("samples")
    plt.show()

def plot_signal_x_time(data: np.ndarray, sampling_rate: int) -> None:
    duration = len(data)/sampling_rate
    step = int(sampling_rate*duration)
    x_time = np.linspace(0, duration, step)
    plt.plot(x_time, data)
    plt.xlabel("time in s")
    plt.show()

def main():
    load_csv_with_pandas()
    # data = convert_out_to_np(Path("embedded_pipeline/out.txt"))
    # sampling_rate = 2000
    # for i, channel in enumerate(data.T):
    #     print(channel.shape)
    #     plot_signal_x_samples(channel)
    #     channel = channel.astype(np.int16)
    #     plot_signal_x_time(channel, sampling_rate)
    #     print(f"data.dtype {channel.dtype}")
    #     save_signal_to_wav(channel, sampling_rate, channel.dtype, Path(f"channel_{i}.wav"))

if __name__ == "__main__":
    main()