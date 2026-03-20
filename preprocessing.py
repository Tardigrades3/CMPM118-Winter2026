import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.utils.class_weight import compute_class_weight
from functions import *

def preprocessing(paths):
    dfs = []
    rep_offset = 0
    for path in paths:
        print(path)
        mat = loadmat(path)

        df = pd.DataFrame(mat['emg'])
        df['stimulus'] = mat['restimulus']
        df['repetition'] = mat['repetition']

        df['repetition'] = df['repetition'] + rep_offset

        rep_offset = df['repetition'].max()

        dfs.append(df)
    # combine all users
    data = pd.concat(dfs, ignore_index=True)

    emg_low = filter_data(data=data, f=20, butterworth_order=4, btype='lowpass')

    emg_notch = notch_filter(data=emg_low,f0=60,Q=30,fs=2000)

    gestures = data['stimulus'].unique().tolist() 
    train_reps, test_reps = train_test_split(
        data['repetition'].unique().tolist(),
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    emg_norm = normalise(data = emg_notch, train_reps=train_reps)

    win_len = 200    
    win_stride = 50

    X_train, y_train, r_train = windowing(emg_norm, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = windowing(emg_norm, test_reps, gestures, win_len, win_stride)
    overlap = np.intersect1d(r_train, r_test)

    if len(overlap) != 0:
        raise Exception(f'repetitions leaked between train and test for repetition {overlap}')
    
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    y_presence_train = (y_train != 0).astype(int)
    y_presence_test = (y_test != 0).astype(int)
    return X_train, y_train, X_test, y_test, y_presence_train, y_presence_test, class_weights_dict