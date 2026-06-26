import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
from sklearn.utils.class_weight import compute_class_weight
import torch

def normalise(data, train_reps):
    """
    Normalise function rewritten to support different numbers of channels
    """
    x = [np.where(data.values[:, data.shape[1] - 2] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :]
    train_data = train_data.reset_index(drop=True)

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

def load_data(path):
    mat = loadmat(path) 
    emg = mat['emg']


    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus']
    data['repetition'] = mat['repetition']
    return data

def extract_td_features(X, n_subwindows=10, zc_threshold=0.01, ssc_threshold=0.01):
    """Convert raw EMG windows into time-domain (Hudgins) feature sequences.

    X : array [N, win_len, C] of windowed (filtered, normalised) EMG.
    Returns [N, n_subwindows, C * 5]: each sub-window summarised by 5 TD features
    per channel — MAV, RMS, waveform length (WL), zero crossings (ZC), slope-sign
    changes (SSC). These are far more robust to amplitude / electrode-placement
    variation than raw samples, which is exactly the cross-subject / cross-rep
    variability that caps held-out accuracy. The RNN still sees a short sequence
    (length n_subwindows) so the recurrent architecture stays meaningful.
    """
    X = np.asarray(X, dtype=np.float64)
    N, win_len, C = X.shape
    sub_len = win_len // n_subwindows
    if sub_len < 2:
        raise ValueError(f"win_len={win_len} too small for n_subwindows={n_subwindows}")

    # Trim to a multiple of n_subwindows, then split the time axis into sub-windows.
    X = X[:, :sub_len * n_subwindows, :]
    Xs = X.reshape(N, n_subwindows, sub_len, C)              # [N, n_sub, sub_len, C]

    mav = np.mean(np.abs(Xs), axis=2)                        # [N, n_sub, C]
    rms = np.sqrt(np.mean(Xs ** 2, axis=2))

    d1 = np.diff(Xs, axis=2)                                 # first difference
    wl = np.sum(np.abs(d1), axis=2)

    s, t = Xs[:, :, :-1, :], Xs[:, :, 1:, :]
    zc = np.sum(((s * t) < 0) & (np.abs(s - t) >= zc_threshold), axis=2)

    d_prev, d_next = d1[:, :, :-1, :], d1[:, :, 1:, :]
    ssc = np.sum(((d_prev * d_next) < 0) &
                 ((np.abs(d_prev) >= ssc_threshold) | (np.abs(d_next) >= ssc_threshold)),
                 axis=2)

    feats = np.stack([mav, rms, wl, zc.astype(np.float64), ssc.astype(np.float64)], axis=-1)
    feats = feats.reshape(N, n_subwindows, C * 5)            # [N, n_sub, C*5]
    return feats.astype(np.float32)


def preprocessing(path, features='raw', n_subwindows=10):
    data = load_data(path)
    return preprocessing_internals(data, features=features, n_subwindows=n_subwindows)

def preprocessing_internals(data, features='raw', n_subwindows=10):
    emg_low = filter_data(data=data, f=[20, 450], butterworth_order=4, btype='bandpass')

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

    if features == 'td':
        X_train = extract_td_features(X_train, n_subwindows=n_subwindows)
        X_test  = extract_td_features(X_test,  n_subwindows=n_subwindows)
        # Standardise TD features (fit on train only) — feature scales differ wildly
        # (WL summed over samples ≫ MAV), and the optimiser converges far better when
        # they're comparable. No test leakage: stats come from train windows only.
        flat = X_train.reshape(-1, X_train.shape[-1])
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0) + 1e-8
        X_train = ((X_train - mu) / sd).astype(np.float32)
        X_test  = ((X_test  - mu) / sd).astype(np.float32)
    elif features != 'raw':
        raise ValueError(f"unknown features='{features}' (expected 'raw' or 'td')")

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))

    return X_train, y_train, X_test, y_test, class_weights_dict

# path = "../NinaProData"

def multi_preprocess(exercise_number, path):
    num_subjects = 27
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [],[]
    for i in range(num_subjects):
        base_path = f"{path}/s{i + 1}/S{i + 1}_A1_E{exercise_number}.mat"
        data = load_data(base_path)
        x_train, y_train, x_test, y_test, _cw = preprocessing_internals(data)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)
    x_test = np.concatenate(x_test_list)
    y_test = np.concatenate(y_test_list)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    x_test = np.transpose(x_test, (0, 2, 1))
    x_train = np.transpose(x_train, (0, 2, 1))
    
    return x_train, y_train, x_test, y_test, class_weights_dict


class NinaProDataset(torch.utils.data.Dataset):
  def __init__(self, x, y):
    self.X = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]