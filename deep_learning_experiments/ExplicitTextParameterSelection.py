"""This experiment explores the use of a CONV1D network using both explicit and implicit text features"""

import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as l
import tensorflow.keras.models as m
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.ops.gen_control_flow_ops import Merge
import pandas as pd
import numpy as np
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler

"""Source: https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""Fetch Explicit Features from File"""
explicitFeatures = pd.read_csv("../data/feature_set_july_20_2.csv")
labels = explicitFeatures["label"].tolist()

explicitFeatures.drop([explicitFeatures.columns[0], explicitFeatures.columns[1]], axis=1, inplace=True)
values = explicitFeatures.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(values,
                                                    labels,
                                                    shuffle=True,
                                                    test_size=0.2)

scaler = MinMaxScaler(feature_range=(0,1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


def build_explicit_features_model(hp):

    model = m.Sequential(name="TestExplicit")
    model.add(l.Input(shape=(x_train.shape[1],)))
    model.add(l.Dense(units=hp.Int(
        'units',
        min_value=64,
        step=32,
        max_value=256
    ), activation='relu', bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.GlorotNormal()))
    model.add(l.Dropout(rate=hp.Choice(
        'rate',
        values=[0.3, 0.4, 0.5]
    )))
    model.add(l.Dense(name="dense2", units=hp.Choice(
        name="units_2",
        values=[32, 64, 128]
    ), activation='relu'))

    model.add(l.Dropout(name="droput2", rate=hp.Choice(
        name="rate_2",
        values=[0.3, 0.4, 0.5]
    )))

    model.add(l.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=k.optimizers.SGD(learning_rate=hp.Choice(
        'learning_rate',
        values=[0.001, 0.005, 0.004],
    )), loss='binary_crossentropy', metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_explicit_features_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=2,
    directory='MIT_2020_FND',
    project_name='text_explicit')

csv_logger = CSVLogger("../data/model_training_7.csv", append=True, separator=';')

tuner.search_space_summary()
tuner.search(x=np.array(x_train_scaled), y=np.array(y_train), epochs=20,
             validation_data=(np.array(x_test_scaled), np.array(y_test)),
             callbacks=[csv_logger])
best_hyperparams = tuner.get_best_hyperparameters(1)
bestModels = tuner.get_best_models(1)
tuner.results_summary()

