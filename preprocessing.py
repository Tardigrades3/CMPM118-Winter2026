import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
import os
from tensorflow import keras as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight



def normalise(data, train_reps):
    """
    Normalise function rewritten to support different numbers of channels
    """
    x = [np.where(data.values[:, data.shape[1] - 2] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :]
    train_data = data.reset_index(drop=True)

    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(train_data.iloc[:, :data.shape[1] - 2])

    scaled = scaler.transform(data.iloc[:, :data.shape[1] - 2])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus']
    normalised['repetition'] = data['repetition']
    return normalised


def filter_data(data, f, butterworth_order=4, btype='lowpass'):
    """
    Filter Data function rewritten to support different numbers of channels
    """
    emg_data = data.values[:, :data.shape[1] - 2]

    f_sampling = 2000
    nyquist = f_sampling / 2
    if isinstance(f, int):
        fc = f / nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i] / nyquist
    b, a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()
    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))
    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus']
    filtered['repetition'] = data['repetition']
    return filtered

def windowing(data, reps, gestures, win_len, win_stride):
    """
    Windowing function rewritten to support different numbers of channels
    """
    if reps:
        x = [np.where(data.values[:, data.shape[1] - 1] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
    if gestures:
        x = [np.where(data.values[:, data.shape[1] - 2] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
    
    idx = [i for i in range(win_len, len(data), win_stride)]
    X = np.zeros([len(idx), win_len, len(data.columns) - 2])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])

    for i, end in enumerate(idx):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:data.shape[1] - 2].values
        y[i] = data.iloc[end, data.shape[1] - 2]
        reps[i] = data.iloc[end, data.shape[1] - 1]

    return X, y, reps


def notch_filter(data, f0, Q, fs=2000):
    """
    Notch Filter function rewritten to support different numbers of channels
    """
    emg_data = data.values[:, :data.shape[1] - 2]

    b, a = signal.iirnotch(f0, Q, fs)
    transpose = emg_data.T.copy()

    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))

    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus'].values
    filtered['repetition'] = data['repetition'].values

    return filtered

def train_model(model, X_train_wind, y_train_wind,
                X_test_wind, y_test_wind,
                save_to, class_weights, epoch=300, ):

    os.makedirs(save_to, exist_ok=True)

    opt_adam = K.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt_adam,
        metrics=['accuracy']
    )

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    mc = ModelCheckpoint(
        os.path.join(save_to, 'best_model.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train_wind, y_train_wind,
        epochs=epoch,
        shuffle=True,
        validation_data=(X_test_wind, y_test_wind),
        callbacks=[es, mc],
        verbose=1,
        class_weight= class_weights
    )

    # Evaluate directly — DO NOT reload
    train_loss, train_acc = model.evaluate(X_train_wind, y_train_wind, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_wind, y_test_wind, verbose=0)

    print(f'Train Accuracy: {train_acc:.3f}')
    print(f'Test Accuracy: {test_acc:.3f}')

    return history, model

def preprocessing(path):
    mat = loadmat(path) 
    emg = mat['emg']

    print(f"subject #{mat['subject']}") # subject number
    print(f"exercise #{mat['exercise']}") # exercise number

    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus']
    data['repetition'] = mat['repetition']

    emg_low = filter_data(data=data, f=20, butterworth_order=4, btype='lowpass')

    emg_notch = notch_filter(data=emg_low,f0=60,Q=30,fs=2000)

    gestures = data['stimulus'].unique().tolist() 
    train_reps, test_reps = train_test_split(
        data['repetition'].unique().tolist(),
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    win_len = 200    
    win_stride = 50

    X_train, y_train, r_train = windowing(emg_notch, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = windowing(emg_notch, test_reps, gestures, win_len, win_stride)
    overlap = np.intersect1d(r_train, r_test)

    if len(overlap) != 0:
        raise Exception(f'repetitions leaked between train and test for repetition {overlap}')
    
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))

    return X_train, y_train, X_test, y_test, class_weights_dict
