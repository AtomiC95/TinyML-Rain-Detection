import wave
import numpy as np

# Open the .wav file
classes = ["heavy_rain", "light_rain", "no_rain", "medium_rain"]
for cls in classes:
    with wave.open(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/wav_data_cropped/{cls}.wav", "rb") as wav_file:
        # Read the number of frames, sample rate, and number of channels
        num_frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        duration = num_frames / float(sample_rate)

        chunk_size = 1

        num_chunks = int(np.ceil(duration / chunk_size))

        for i in range(num_chunks):
            # Calculate the start and end frames for the chunk
            start_frame = int(i * chunk_size * sample_rate)
            end_frame = min(int((i + 1) * chunk_size * sample_rate), num_frames)

            # Set the position in the .wav file to start_frame
            wav_file.setpos(start_frame)

            # Read the audio data for the chunk
            raw_audio_data = wav_file.readframes(end_frame - start_frame)

            # Convert the raw audio data to a numpy array
            audio_data = np.frombuffer(raw_audio_data, dtype=np.int16)

            # Trim the numpy array to the desired chunk size
            trimmed_audio_data = audio_data[:int(chunk_size * sample_rate)]

            # Convert the trimmed numpy array back to raw audio data
            trimmed_raw_audio_data = trimmed_audio_data.tobytes()

            # Write the trimmed audio data to a new .wav file
            with wave.open(f"/home/nikolas/git/TinyML-Rain-Detection/model_creation/model_data/train_data/{cls}/chunk_{i+1}.wav", "wb") as chunk_file:
                chunk_file.setnchannels(num_channels)
                chunk_file.setsampwidth(wav_file.getsampwidth())
                chunk_file.setframerate(sample_rate)
                chunk_file.writeframes(trimmed_raw_audio_data)