import os
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import tqdm

DATA_PATH = r"DataSet\DATASET_SUPER"
GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
def extract_features_single_file(file_info):
    file_path, label = file_info
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate((mfcc, delta1, delta2), axis=0).T
        if len(combined) >= 430: combined = combined[:430, :]
        else: combined = np.pad(combined, ((0, 430 - len(combined)), (0, 0)))
        return combined, label
    except:
        return None
if __name__ == "__main__":
    file_list = []
    for idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATA_PATH, genre)
        if os.path.exists(genre_dir):
            for f in os.listdir(genre_dir):
                if f.endswith(".wav"):
                    file_list.append((os.path.join(genre_dir, f), idx))
    print(f" Extracting 120 Features from {len(file_list)} files...")
    X, y = [], []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(extract_features_single_file, file_list), total=len(file_list)))
    for res in results:
        if res is not None:
            X.append(res[0])
            y.append(res[1])
    X = np.array(X)
    y = np.array(y)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    np.save("NpyData/scaling_params.npy", {'mean': mean, 'std': std})
    np.save("NpyData/X_mfcc_UP.npy", X)
    np.save("NpyData/y_mfcc_UP.npy", y)
    print(f"\n Files saved: X_mfcc_UP.npy, y_mfcc_UP.npy, scaling_params.npy")