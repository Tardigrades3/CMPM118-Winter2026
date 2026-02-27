
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from preprocessing import *
from sklearn.metrics import classification_report, confusion_matrix

# directory to save model to (as a full path)
save_to = './model'

model = Sequential([
    Input(shape=(200, 10)),   # (time, channels)

    Conv1D(32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(13, activation='softmax')
])

X_train, y_train, X_test, y_test, class_weights = preprocessing('s1/S1_A1_E1.mat')

# insert your model here! 
history, saved_model = train_model(model, X_train, y_train, X_test, y_test, save_to= save_to,class_weights= class_weights, epoch=30, )

y_pred = np.argmax(saved_model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))