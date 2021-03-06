from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import LSTM

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
#keras db already takes care of the pre-processing
#this takes really long to train

x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)

model = Sequential()
model.add(Embedding(20000, 128)) #tensor vector of fixed size - 128 neurons
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          verbose=2,
          validation_data=(x_test, y_test))