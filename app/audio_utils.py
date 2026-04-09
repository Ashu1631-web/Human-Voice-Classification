import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc, pitch, zcr]).reshape(1, -1)
