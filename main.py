
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from preprocessing import *
from functions import *
from sklearn.metrics import classification_report, confusion_matrix
from model import *
import numpy as np

# directory to save model to (as a full path)
save_to = '/Users/tyham/Documents/CMPM118-Winter2026/model'

#initial training set
training_set = [
    's1/S1_A1_E1.mat',
    's2/S2_A1_E1.mat',
    's3/S3_A1_E1.mat',
    's4/S4_A1_E1.mat',
    's5/S5_A1_E1.mat',
    's6/S6_A1_E1.mat',
]
X_train, y_train, X_test, y_test, yp_train, yp_test, class_weights = preprocessing(training_set)

sq = sequential()
cb = combined_model()

# insert your model here! 
history, saved_model = train_model(sq ,X_train, y_train, X_test, y_test, save_to= save_to,class_weights= class_weights, epoch=30, )


y_pred = np.argmax(saved_model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))



cb.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

y_pred2 = np.argmax(cb.predict(
    X_test,
), axis=1)

print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred2))
