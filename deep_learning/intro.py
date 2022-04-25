import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

num_classes = 10 # numbers 0 to 9
num_featueres = 784  # image pixels (28x28)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_featueres]), x_test.reshape([-1, num_featueres])

x_train, x_test = x_train / 255., x_test / 255.

# define hper-params
learning_rate = 0.001
training_steps = 3000
batch_size = 250
display_step = 100
n_hidden = 512  # num hidden neurons

# shuffle and batch the data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(batch_size=batch_size).prefetch(1)

# initialize weights and biases
random_normal = tf.initializers.RandomNormal()
weights = {
    "h": tf.Variable(random_normal([num_featueres, n_hidden])),
    "out": tf.Variable(random_normal([n_hidden, num_classes]))
}

biases = {
    "b": tf.Variable(tf.zeros([n_hidden])),
    "out": tf.Variable(tf.zeros([num_classes]))
}

# define the loss function (cross entropy): - penalizes incorrect predictions
def cross_entropy(y_predict, y_true):
    # convert label to one-hot vector [0,0,0,0,1,0,0,0]...
    y_true = tf.one_hot(y_true, depth=num_classes)
    # avoid log 0 error
    y_predict = tf.clip_by_value(y_predict, 1e-9, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_predict)))

def run_optimizer(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)

        train_variables = list(weights.values()) + list(biases.values())
        optimizer = tf.keras.optimizers.SGD(learning_rate)  # ways to train this
        gradients = g.gradient(loss, train_variables)
        optimizer.apply_gradients(zip(gradients, train_variables))


def neural_net(inputData):
    #inputData (features) * weights + bias
    hidden_layer = tf.add(tf.matmul(inputData, weights["h"]), biases["b"])
    #apply the activation function (sigmoid function) at the layer
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    #output layer repeats weights + biases
    out_layer = tf.matmul(hidden_layer, weights["out"]) + biases["out"]
    #softmax noramlized the outputs to a probabily items belong to a certain class
    return tf.nn.softmax(out_layer)


def calculate_accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

#perform the training of neural network
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimizer(batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        accuracy = calculate_accuracy(pred, batch_y)
        print("Training epoch: " + str(step) + "; Loss: " + str(loss) + "; Accuracy: " + str(accuracy))

#test the neural network:
predicted_labels = neural_net(x_test)
print("Accuracy: " + str(calculate_accuracy(predicted_labels, y_test)))