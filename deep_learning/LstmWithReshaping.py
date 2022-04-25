from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense, LeakyReLU
import pandas as pd
import numpy as np
from tensorflow.keras import activations
from PreProcessingUtils import PreProcessingUtils
import tensorflow as tf
from sklearn.model_selection import train_test_split

devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

feature_file = pd.read_csv("../data/feature_set_with_doc_vectors_july_final.csv")

labels = np.array(feature_file["label"])
features = feature_file.iloc[:, 1:]
columns = ["text", "title"]
feature_utils = PreProcessingUtils(dataFrame=feature_file, columnNames=columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(features)

reshapedArray = feature_utils.get_reshaped_features(np.array(scaledData))
#create test/train split
x_train, x_test, y_train, y_test = train_test_split(reshapedArray, labels, test_size=0.3)

model = Sequential(name="RNN_LSTM_1")
model.add(LSTM(512, batch_input_shape=(x_train.shape[0], 1, x_train.shape[2]),
               stateful=True,
               return_sequences=True,
               name="input_lstm_1",
               ))
model.add(Dropout(0.3))
model.add(LSTM(units=384, return_sequences=True, name="input_lstm_2", activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(units=64, return_sequences=False, name="input_lstm_3", activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=2, activation=activations.softmax))
model.compile(optimizer='adam', metrics=['accuracy'],  loss='binary_crossentropy')
model.summary()
lstmHistory = model.fit(x=x_train, y=y_train, epochs=30, shuffle=True, workers=4,
                      use_multiprocessing=True, batch_size=x_train.shape[0])

y_predictions = model.predict(x_test, use_multiprocessing=True, workers=4)
#do predictions
