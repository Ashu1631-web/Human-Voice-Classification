import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    features = []

    # Spectral features
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.rms(y=y)))

    # Pitch
    pitch = librosa.yin(y, fmin=20, fmax=300)
    features.append(np.mean(pitch))

    # MFCC (Top 10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    for i in range(10):
        features.append(np.mean(mfcc[i]))

    return np.array(features).reshape(1, -1)
