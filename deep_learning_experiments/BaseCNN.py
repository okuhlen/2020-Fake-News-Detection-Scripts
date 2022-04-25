import IPython
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.layers import Flatten, Input, MaxPool1D, Conv1D, Embedding, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np
import PreProcessingUtils as p
import documents.DocumentEmbeddingUtils as d
import tensorflow as tf
import tensorflow.keras.initializers as k

"""Experiment 1: Demonstrate the use of a base CNN and word embeddings (word2vec). Seems like everyone is doing it 
this way. PANDAS == slow trash"""
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        IPython.display.clear_output(wait=True)

documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv", chunksize=5000, error_bad_lines=False, sep=",")
special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame() #concattenated title and body
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

x_train, x_test, y_train, y_test = train_test_split(encoded_documents, labels, test_size=0.3)

#Using the embeddings, vocab size, feed this information into a keras model - Todo: Test params.
def run_cnn_training():
    # Using the embeddings, vocab size, feed this information into a keras model - Todo: Test params.
    cnn_input_model = Input(shape=(encoded_documents.shape[1],), name="InputLayer")

    model_one = Embedding(len(unique_words_array),
                           300,
                           input_length=maximum_words,
                           trainable=False,
                           weights=[document_word_embeddings],
                          name="EmbeddingLayer")(cnn_input_model)

    model_one = Conv1D(filters=70, kernel_size=3, strides=4, kernel_initializer=k.GlorotNormal(),
                       bias_initializer=k.Zeros(),
                       name="Conv1DOne", activation='relu')(model_one)

    model_one = MaxPool1D(pool_size=2, name="MaxPoolOne")(
        model_one)

    model_one = Dropout(rate=0.5, name="DropoutOne")(model_one)

    model_one = Conv1D(filters=70, kernel_size=4, strides=4, activation='relu',
        kernel_initializer=k.GlorotNormal(),
        bias_initializer=k.Zeros(),
        name="DropoutTwo")(model_one)

    model_one = MaxPool1D(pool_size=3, name="MaxPoolTwo")(
        model_one)

    #model_one = Dropout(rate=hp.Choice(name="dropout_last_layer_rate",values=[0.3, 0.4, 0.5]))(model_one)

    #output_layer = Dense(units=hp.Choice(name="cnn_final_dense_layer", values=[16, 32, 64]), activation="relu")(
        #model_one)
    model_one = Dropout(rate=0.3, name="FinalDropout")(model_one)

    model_one = Flatten(name="FlattenLayer")(model_one)

    output_layer = Dense(name="DenseLayerFinal", units=64, activation="relu")(model_one)

    output_layer = Dense(units=1, activation='sigmoid', name="OutputLayer")(output_layer)

    model = Model(inputs=[cnn_input_model], outputs=[output_layer], name="BaseCNN")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

                    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                    tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives()],

                    loss='binary_crossentropy')

    plot_model(model, to_file="../data/cnn_model_experiment.png", show_shapes=True,
               show_layer_names=True)

    model.summary()

    csv_logger = CSVLogger("../data/cnn_experiment_result_data.csv", append=True, separator=';')

    model.fit(np.asarray(x_train), np.asarray(y_train),
                                  validation_data=(np.asarray(x_test), np.asarray(y_test)),
                                  epochs=20,
                                  callbacks=[csv_logger])

run_cnn_training()