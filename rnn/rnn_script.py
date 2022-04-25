import tensorflow.keras as k
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as op
import numpy as np
import pandas as pd
from docutils.io import Input
from sklearn.model_selection import train_test_split

featureDataSet = pd.read_csv("../data/feature_set_with_doc_vectors_july_final.csv", low_memory=False, sep=",")
featureDataSet.drop([featureDataSet.columns[0], featureDataSet.columns[1]], axis=1, inplace=True)
values = featureDataSet.iloc[:, 1:]
batchSize = 2000

print(values.shape)
labels = featureDataSet["label"].tolist()
values = featureDataSet.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(values, labels, train_size=0.8, test_size=0.2)
transformed_x_train = np.array(x_train)
transformed_x_test = np.array(x_test)
transformed_y_train = np.array(y_train)
transformed_y_test = np.array(y_test)

dim3_x_train = np.reshape(transformed_x_train, (1,) + transformed_x_train.shape )
dim3_x_test = np.reshape(transformed_x_test, (1,) + transformed_x_test.shape )

print(dim3_x_train)
print(transformed_x_test.shape[1])

model = km.Sequential()

inputLayer = k.Input(shape=(transformed_y_train.shape[1], ), name="doc2vec_inputs")
model.add(kl)
#model.add(kl.Embedding(input_shape=(transformed_x_train.shape[1], 1), input_length=transformed_x_train.shape[1],
                       #input_dim=transformed_x_train.shape[1], output_dim=1))
model.add(kl.Dropout(0.2))
model.add(kl.LSTM(units=384, activation='relu', recurrent_activation='sigmoid'))
model.add(kl.Dropout(0.2))
model.add(kl.LSTM(units=128, activation='relu', recurrent_activation='sigmoid'))
model.add(kl.Dropout(0.2))
model.add(kl.LSTM(units=32, activation='relu', recurrent_activation='sigmoid'))
model.add(kl.Dense(activation='softmax', units=2))

# use binary cross-entropy loss function (good for two labels).
opt = op.SGD(learning_rate=0.001)
model.compile(optimizer=opt, metrics=['accuracy'], loss='binary_crossentropy')

model.fit(x=dim3_x_train, y=transformed_y_train, batch_size=2000, epochs=20, shuffle=True)

print(model.summary())

y_pred = model.predict(transformed_x_test)
