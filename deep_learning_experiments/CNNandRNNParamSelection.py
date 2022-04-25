import tensorflow as tf
import kerastuner as kt
import tensorflow.keras.layers as l
import pandas as pd
from kerastuner import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.utils.vis_utils import plot_model

from PreProcessingUtils import PreProcessingUtils
import numpy as np
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils
import tensorflow.keras.models as m
import graphviz as g

"""Explicit features"""
explicitFeatures = pd.read_csv("../data/feature_set_july_20_2.csv")
labels = explicitFeatures["label"].tolist()
explicitFeatures.drop([explicitFeatures.columns[0], explicitFeatures.columns[1]], axis=1, inplace=True)
values = explicitFeatures.iloc[:, 1:]

"""Implicit Text Features: Basic pre-processing, create CNN model."""
documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv", chunksize=5000, error_bad_lines=False, sep=",")
special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame()
simplifiedDataFrame = pd.DataFrame()
for i in documents:
    pp = PreProcessingUtils(i, ["title", "text"])
    pp.remove_stop_words()
    pp.remove_special_characters(special_characters)
    temp = pp.get_data_frame()
    simplifiedDataFrame = pd.concat([simplifiedDataFrame, pp.concatenate_title_and_body()])
    cleaned_documents = pd.concat([cleaned_documents, pp.get_data_frame()])

rawDocumentLabels = np.asarray(cleaned_documents["label"].tolist())
deu = DocumentEmbeddingUtils(simplifiedDataFrame)
maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

imp_x_train, imp_x_test, imp_y_train, imp_y_test = train_test_split(encoded_documents,
                                                                    labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2)

exp_x_train, exp_x_test, exp_y_train, exp_y_test = train_test_split(values,
                                                                    labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2)


def build_rnn_and_cnn_model(hp):
    input_layer = l.Input(name="InputOne",
                          shape=(encoded_documents.shape[1],))

    embeddings = l.Embedding(len(unique_words_array),
                             300,
                             trainable=False,
                             weights=[document_word_embeddings],
                             name="EmbeddingOne")(input_layer)

    conv_1d_layer = l.Conv1D(name="Conv1D",
                             filters=hp.Choice(
                                 name="filters",
                                 values=[5, 10, 15, 20],
                             ),
                             kernel_size=hp.Choice(
                                 name="kernel_size",
                                 values=[10, 15, 20, 25, 30]
                             ),
                             padding="same", activation='relu')(embeddings)

    max_pool_1d = l.MaxPool1D(name="MaxPoolOne",
                              pool_size=hp.Choice(
                                  name="pool_size",
                                  values=[2, 3, 4]
                              ),
                              strides=2)(conv_1d_layer)

    max_pool_1d = l.Dropout(name="Dropout1",
                            rate=hp.Choice(name="rate",
                                           values=[0.2, 0.3, 0.4, 0.5]))(max_pool_1d)

    lstm_layer = l.LSTM(name="LSTMOne", activation=tf.keras.activations.tanh,
                        units=hp.Choice(name="lstm_units",
                                        values=[128, 256, 384, 512]),
                        stateful=False,
                        return_sequences=False,
                        recurrent_dropout=0.,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        recurrent_activation=tf.keras.activations.sigmoid,
                        use_bias=True)(max_pool_1d)

    lstm_layer = l.Dropout(rate=hp.Choice(
        "dropout_1",
        values=[0.3, 0.4, 0.5]))(lstm_layer)

    lstm_layer = l.Dense(units=hp.Choice(
        name="lstm_one",
        values=[128, 160, 192]
    ), activation=tf.keras.activations.relu)(lstm_layer)

    lstm_layer = l.Flatten()(lstm_layer)

    output_layer = l.Dense(units=1, activation=tf.keras.activations.sigmoid)(lstm_layer)

    finalModel = m.Model(inputs=[input_layer], outputs=[output_layer], name="RNN_CNN")

    finalModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice(
        name="learningRate",
        values=[0.001, 0.005]
    )), metrics=['accuracy'],
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    plot_model(finalModel, to_file="../data/hybrid_cnn_rnn.png",
               show_shapes=True, show_dtype=False,
               show_layer_names=True)

    finalModel.summary()

    plot_model(model=finalModel, to_file="../data/hybrid_cnn_rnn.png",
               show_shapes=True, show_layer_names=True, show_dtype=False)
    return finalModel


tuner = RandomSearch(
    build_rnn_and_cnn_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=2,
    directory="mit_rnn_cnn_dnn",
    project_name="fake_news_detection2"
)
tuner.search_space_summary()

csvFile = CSVLogger(filename="../data/hybrid_cnn_and_rnn.csv", separator=";", append=False)
tuner.search(x=np.asarray(imp_x_train),
             y=np.asarray(imp_y_train), callbacks=[csvFile],
             validation_data=(np.asarray(imp_x_test), np.asarray(imp_y_test)),
             epochs=50,
             batch_size=64)

bestModels = tuner.get_best_models(1)
tuner.results_summary()
