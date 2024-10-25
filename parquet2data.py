from underthesea import word_tokenize
from vinorm import TTSnorm
import os

import pandas as pd
import wave
import io

total_time = 0
wave_save_dir = "infore_25h_training_data/wavs"
text_file_path = "infore_25h_training_data/texts/training_texts_file.txt"
training_texts_file = open(text_file_path, "w+", encoding="utf-8")

for parquet_fname in os.listdir("infore_25h"):
    # Read Parquet file using Pandas
    print(parquet_fname)
    df = pd.read_parquet(f"infore_25h/{parquet_fname}")

    for index, row in df.iterrows():
        audio_data = row["audio"]  # get audio data from row "audio"
        audio_bytes = audio_data["bytes"]  # get audio bytes from audio data
        fname = audio_data["path"]  # get audio file name
        transcript = row["transcription"]  # get transcript 
        try:
            # convert audio bytes to file-like object
            audio_file = io.BytesIO(audio_bytes)

            # save audio file to disk
            with wave.open(f"{wave_save_dir}/{fname}", "wb") as wf:
                # read audio bytes and write to wave file
                # You need to know the parameters of the original audio file (sample rate, sample width, number of channels).
                wf.setparams(
                    (1, 2, 44100, 0, "NONE", "not compressed")
                )  # These params can change depend on the original audio file
                wf.writeframes(audio_file.read())
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                total_time += duration
                
            norm_text = TTSnorm(transcript, lower=False)
            word_segment = word_tokenize(norm_text, format="text")
            training_texts_file.write(f"DUMMY/{fname}|{word_segment}\n")

        except Exception as e:
            print(f"Lỗi khi xử lý file {index}: {e}")

print("Total time:", total_time)

'''
If you want down sample rate of audio file to 22050Hz, you can use the following code:

from pydub import AudioSegment
import os

for fname in os.listdir("infore_25h_training_data/wavs"):
    sound = AudioSegment.from_file("infore_25h_training_data/wavs/" + fname)
    if sound.frame_rate != 44100:
        print(fname, sound.frame_rate)
    # print(fname, sound.frame_rate)
    new_sr = sound.frame_rate // 2
    sound_resampled = sound.set_frame_rate(new_sr)
    sound_resampled.export("infore_25h_training_data/new_wavs/" + fname, format="wav")
'''
