from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout
from tensorflow.keras import Sequential
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten
from PreProcessingUtils import PreProcessingUtils
import numpy as np
import kerastuner as kt

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
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(reshapedArray)
x_train, y_train, x_test, y_test = train_test_split(reshapedArray)
model = Sequential(name="CNN_Model")
model.add(Conv1D(name="conv_layer_1",
                 input_shape=(scaledData.shape[0], 1, scaledData[2]),
                 activation='relu',
                 kernel_size=3, filters=32))
model.add(MaxPool1D(pool_size=3))
model.add(Dropout(rate=0.2))
model.add(Conv1D(name="conv_layer_2",
                 activation="relu",
                 kernel_size=3,
                 filters=32))
model.add(MaxPool1D(pool_size=32))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=2, activation='softmax'))
model.summary()
model.compile(optimizer="adam", metrics=["accuracy"])

y = model.fit()

