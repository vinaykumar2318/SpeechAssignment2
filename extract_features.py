import os
import librosa
import numpy as np
import pandas as pd

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def load_dataset(folder_path, label):
    clips_path = os.path.join(folder_path, "clips")
    tsv_path = os.path.join(folder_path, "validated.tsv")

    df = pd.read_csv(tsv_path, sep='\t')

    print("Columns found:", df.columns)

    if "path" in df.columns:
        file_column = "path"
    elif "audio_file" in df.columns:
        file_column = "audio_file"
    elif "filename" in df.columns:
        file_column = "filename"
    else:
        raise Exception("No valid audio file column found!")

    X = []
    y = []

    for file_name in df[file_column][:100]:
        file_path = os.path.join(clips_path, str(file_name))

        if os.path.exists(file_path):
            mfcc = extract_mfcc(file_path)

            for frame in mfcc:
                X.append(frame)
                y.append(label)

    return X, y