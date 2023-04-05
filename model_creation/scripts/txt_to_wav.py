import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile

def load_csv_with_pandas():
    classes = ["heavy_rain", "light_rain", "no_rain", "medium_rain"]
    for cls in classes:
        with open(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/raw_data/{cls}.txt", 'r') as f:
            data_str = f.read()

        # Step 2: convert data to numpy array
        data_list = data_str.strip().split('\n')
        data = np.array([int(d) for d in data_list])

        # Step 3: set sample rate
        sample_rate = 8000

        # Step 4: convert to appropriate data type and write to WAV file
        write(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data/{cls}.wav", sample_rate, data.astype(np.int16))


def crop_file():
    classes = ["heavy_rain", "light_rain", "no_rain", "medium_rain"]
    for cls in classes:
        sr, signal = wavfile.read(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data/{cls}.wav")

        # specify the start and end time in seconds
        start_time = 5.0
        end_time = 165.0
        sample_rate = 8000
        # convert the start and end time to sample indices
        start_index = int(start_time * sample_rate)
        end_index = int(end_time * sample_rate)

        # extract the desired portion of the signal
        cropped_signal = signal[start_index:end_index]

        # write the cropped signal to a new file
        wavfile.write(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/{cls}.wav", sample_rate, cropped_signal)




def main():
    load_csv_with_pandas()
    crop_file()

if __name__ == "__main__":
    main()
