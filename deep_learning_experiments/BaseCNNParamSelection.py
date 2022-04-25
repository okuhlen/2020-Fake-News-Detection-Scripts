import IPython as IPython
import kerastuner as kt
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, Dropout, Dense
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Flatten, Input
import tensorflow.keras as k
from tensorflow.python.keras.utils.vis_utils import plot_model

import PreProcessingUtils as p
import documents.DocumentEmbeddingUtils as d
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        IPython.display.clear_output(wait=True)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv", chunksize=5000, error_bad_lines=False, sep=",")

special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame()  # concattenated title and body
simplifiedDataFrame = pd.DataFrame()
for i in documents:
    pp = p.PreProcessingUtils(i, ["title", "text"])
    pp.remove_stop_words()
    pp.remove_special_characters(special_characters)
    temp = pp.get_data_frame()
    simplifiedDataFrame = pd.concat([simplifiedDataFrame, pp.concatenate_title_and_body()])
    cleaned_documents = pd.concat([cleaned_documents, pp.get_data_frame()])

labels = cleaned_documents["label"].tolist()
# Get the vocab, encoded dataset and document embeddings
deu = d.DocumentEmbeddingUtils(simplifiedDataFrame)
maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

x_train, x_test, y_train, y_test = train_test_split(encoded_documents, labels, test_size=0.2, shuffle=True)

def test_model(hp):
    """The purpose of this function is to build a model that Keras-Tuner uses to test """
    cnn_input_model = Input(shape=(encoded_documents.shape[1],), name="InputLayer")

    model_one = Embedding(len(unique_words_array),
                           300,
                           input_length=maximum_words,
                           trainable=False,
                           weights=[document_word_embeddings],
                          name="EmbeddingLayer")(cnn_input_model)

    model_one = Conv1D(filters=hp.Choice(name="conv_layer_one_filters", values=[70, 90, 110]), kernel_size=hp.Choice(
        name="conv_layer_one_kernel", values=[2, 3, 4]), activation='relu', strides=hp.Choice(
                           name="conv_layer_one_strides",
                           values=[2, 3, 4]),
                       kernel_initializer=k.initializers.GlorotNormal(),
                       bias_initializer=k.initializers.Zeros(),
                       name="Conv1DOne")(model_one)

    model_one = MaxPool1D(pool_size=hp.Choice(name="max_pool_one_pool_size", values=[2, 3, 4]), name="MaxPoolOne")(
        model_one)

    model_one = Dropout(rate=hp.Choice(name="dropout_one_rate", values=[0.3, 0.4, 0.5]), name="DropoutOne")(model_one)

    model_one = Conv1D(
        filters=hp.Choice(name="conv_layer_two_filters", values=[30, 50, 70]), kernel_size=hp.Choice(
            name="conv_layer_two_kernel",
            values=[2, 3, 4]
        ),
        activation='relu',

        strides=hp.Choice(
            name="conv_layer_two_strides",
            values=[2, 3, 4]
        ),
        kernel_initializer=k.initializers.GlorotNormal(),
        bias_initializer=k.initializers.Zeros(),
        name="DropoutTwo")(model_one)

    model_one = MaxPool1D(pool_size=hp.Choice(name="max_pool_two_pool_size", values=[2, 3, 4, 5]), name="MaxPoolTwo")(
        model_one)

    model_one = Dropout(rate=hp.Choice(name="output_droupout", values=[0.3, 0.4, 0.5]), name="FinalDropout")(model_one)

    model_one = Flatten(name="FlattenLayer")(model_one)

    output_layer = Dense(name="DenseLayerFinal", units=hp.Choice(name="dense_final_layer_units",
                                                                 values=[16, 32, 64]), activation="relu")(model_one)

    output_layer = Dense(units=1, activation='sigmoid', name="OutputLayer")(output_layer)

    model = Model(inputs=[cnn_input_model], outputs=[output_layer], name="BaseCNN")

    model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=hp.Choice(
                    name="learning_rate",
                    values=[0.001, 0.0001, 0.0005])),

                    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                    tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives()],

                    loss='binary_crossentropy')

    plot_model(model, to_file="../data/cnn_experiment_model_graph_trial_two.png", show_shapes=True,
               show_layer_names=True)

    model.summary()

    return model


model_training_results = kt.RandomSearch(test_model,
                                         objective='val_accuracy',
                                         directory='CNN_Implicit_2',
                                         project_name='cnn_implicit_2',
                                         max_trials=3,
                                         executions_per_trial=2)

csv_logger = CSVLogger("../data/cnn_base_data_file_2.csv", append=True, separator=';')

model_training_results.search(np.asarray(x_train), np.asarray(y_train),
                              validation_data=(np.asarray(x_test), np.asarray(y_test)),
                              epochs=20,
                              callbacks=[csv_logger])

best_params = model_training_results.get_best_hyperparameters(1)
bestModel = model_training_results.get_best_models(1)
model_training_results.results_summary()
print(best_params)
