from kerastuner import HyperParameter
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
import tensorflow.keras as k

devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
feature_file = pd.read_csv("../data/feature_set_with_doc_vectors_july_final.csv")
labels = np.array(feature_file["label"])
features = feature_file.iloc[:, 1:]
columns = ["text", "title"]
feature_utils = PreProcessingUtils(dataFrame=feature_file, columnNames=columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(features)

# x_train, x_test, y_train, y_test = train_test_split(np.array(scaledData), labels, test_size=0.3)
reshapedArray = feature_utils.get_reshaped_features(np.array(scaledData))
x_train, x_test, y_train, y_test = train_test_split(reshapedArray, labels, test_size=0.3)


# create a function which accepts an object which is used to define hyper-parameter search
# bands for parameters of a given model
def create_model(hp):
    model = Sequential(name="CNN_Model")
    model.add(Conv1D(name="conv_layer_1",
                     input_shape=(1, x_train.shape[2]),
                     activation=hp.Choice("activation",
                                          values=["relu", "tanh"]),
                     kernel_size=1,
                     use_bias=True,
                     kernel_initializer=k.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                     bias_initializer=k.initializers.Zeros(),
                     trainable=True,
                     filters=hp.Int("filters",
                                    min_value=5,
                                    max_value=32,
                                    )))
    model.add(MaxPool1D(pool_size=1, name="maxpooling_layer1"))
    model.add(Dropout(rate=hp.Choice("rate",
                                     values=[0.1, 0.2, 0.3]
                                     )))
    model.add(Conv1D(name="conv_layer_2",
                     activation=hp.Choice("activation",
                                          values=["relu", "selu", "tanh"]),
                     kernel_size=1,
                     kernel_initializer=k.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                     bias_initializer=k.initializers.Zeros(),
                     filters=32))
    model.add(MaxPool1D(pool_size=hp.Choice(
        values=[1, 2, 3, 4]
    ), name="maxpooling_layer2"))
    model.add(Dropout(rate=hp.Int
    ("rate",
     min_value=0.1,
     max_value=0.3)))
    # model.add(Flatten())
    model.add(Dense(units=hp.Int(
        "units",
        min_value=32,
        max_value=128
    ), activation='relu'))
    model.add(Dropout(rate=hp.Int
    ("rate",
     min_value=0.1,
     max_value=0.3)))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(optimizer=k.optimizers.Adam(
        hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]),
    ), metrics=["accuracy"], loss="binary_crossentropy")
    return model


tuner = kt.RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='MIT_2020_FND',
    project_name='MIT_2020_FND'
)

tuner.search(x_train, y_train, validation_data=(x_test, y_test))
tuner.results_summary()
