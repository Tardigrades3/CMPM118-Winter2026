import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date, time
import torch
import snntorch as snn
import pandas as pd

def preprocess_emg(emg_data, fs=2000, filter_order=4):
    """Preprocess EMG for SNN"""
    # Bandpass filter (20-500 Hz)
    sos = butter(filter_order, [20, 500], btype='band', fs=fs, output='sos')
    emg_filtered = filtfilt(sos[0], sos[1], emg_data, axis=0)
    
    # Normalize per channel
    scaler = StandardScaler()
    emg_normalized = scaler.fit_transform(emg_filtered)
    
    return emg_normalized

def encode_to_spikes(emg_data, threshold_percentile=75):
    """Convert EMG to spike train (binary)"""
    threshold = np.percentile(np.abs(emg_data), threshold_percentile)
    spikes = (np.abs(emg_data) > threshold).astype(np.float32)
    return spikes

# load mat-file
mat = loadmat('s1/S1_A1_E1.mat')  

#print the keys of the loaded mat-file to understand its structure
print(mat.keys())
print(f"subject: #{mat['subject']}") # subject number
print(f"exercise: #{mat['exercise']}") # exercise number
print(f"stimulus: {mat['stimulus']}") # Original movement label
print(f"restimulus: {mat['restimulus']}") # Corrected movement label
print(f"repetition: {mat['repetition']}") # Trial number of original stimulus
print(f"rerepetition: {mat['rerepetition']}") # Trial number after relabeling
print(f"emg:\n {mat['emg']}") # EMG data

emg = mat['emg']
emg_prep = preprocess_emg(emg)
spike_train = encode_to_spikes(emg_prep)
print(f"Spike train shape: {spike_train.shape}")

#emg shape is (101014,10)


