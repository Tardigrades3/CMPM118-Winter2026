from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense, Multiply, Lambda)
from preprocessing import *
from functions import *
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model

def sequential():
    return Sequential([
    Input(shape=(200, 10)),   # (time, channels)

    Conv1D(32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(13, activation='softmax')
])

def binary_classifier():
    return Sequential([
    Input(shape=(200,10)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

def combined_model():
    inputs = Input(shape=(200, 10))  # (time, channels)

    # --- Stage 1: Logistic Regression for signal presence ---
    # Flatten input for logistic regression (treats it as a flat feature vector)
    flat_input = Flatten()(inputs)
    signal_present = Dense(1, activation='sigmoid', name='presence')(flat_input)

    # --- Stage 2: 1D CNN for signal classification ---
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    cnn_logits = Dense(13, activation='softmax')(x)  # raw classification

    # --- Gating: zero out CNN output if no signal detected ---
    # signal_present (B,1) gates cnn_logits (B,13)
    gated_type = Multiply(name='type')([cnn_logits, signal_present])

    return Model(inputs=inputs, outputs=gated_type)