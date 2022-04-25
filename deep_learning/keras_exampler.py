import tensorflow.keras.optimizers
from tensorflow.keras.datasets import mnist
import tensorflow.keras.datasets as ds
import tensorflow.keras as k
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.models as m
import tensorflow.keras.layers as l

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#flatten the data to 1D, from test train split - reshape is (y, x).
transformed_train_images = train_images.reshape(60000, 784)
transformed_test_images = test_images.reshape(10000, 784)
transformed_train_images = transformed_train_images.astype('float32')
transformed_test_images = transformed_test_images.astype('float32')
#normalize the image data by dividing by 255
transformed_test_images /= 255
transformed_train_images /= 255

#transform data to one-hot encoded data
transformed_train_labels = k.utils.to_categorical(train_labels, 10) #labels, and num classes
transformed_test_labels = k.utils.to_categorical(test_labels, 10)

#create the model
model = m.Sequential()
model.add(l.Dense(activation='relu', input_shape=(784,), units=512))
model.add(l.Dense(activation='relu', units=128))
model.add(l.Dense(activation='softmax', units=10))

#setup the loss function and optimization function
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
result = model.fit(transformed_train_images,
                   transformed_train_labels,
                   batch_size=100,
                   epochs=10,
                   verbose=2,
                   validation_data=(
    transformed_test_images, transformed_test_labels))

#get a summary of the model.
model.summary()
