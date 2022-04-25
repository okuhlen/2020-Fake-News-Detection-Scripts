import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.utils.vis_utils import plot_model
import kerastuner as kt
import numpy as np
from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Embedding
from tensorflow.keras.models import Model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

documents = pd.read_csv("../data/summarized_text_11.csv", error_bad_lines=False, sep=",")

special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame()  # concattenated title and body
simplifiedDataFrame = pd.DataFrame()

rows_to_remove = []
num_null_rows = 0

for index, row in documents.iterrows():
    if row.isnull()["text"] or row.isnull()["label"]:
        rows_to_remove.append(row["id"])
        num_null_rows = num_null_rows + 1
        print("remove nan done at " +str(num_null_rows))
    print("Done with " +str(index))

trimmed_documents = documents.drop(index=rows_to_remove)
print(trimmed_documents.shape)

pp = PreProcessingUtils(trimmed_documents, ["text"])
pp.remove_stop_words()
pp.remove_special_characters(special_characters)
temp = pp.get_data_frame()
simplifiedDataFrame = pd.concat([simplifiedDataFrame, temp])
cleaned_documents = pd.concat([cleaned_documents, pp.get_data_frame()])

#for i in documents:
    #pp = PreProcessingUtils(i,["title", "text"])
    #pp.remove_stop_words()
    #pp.remove_special_characters(special_characters)
    #temp = pp.get_data_frame()
    #simplifiedDataFrame = pd.concat([simplifiedDataFrame, pp.concatenate_title_and_body()])
   # cleaned_documents = pd.concat([cleaned_documents, temp])

labels = cleaned_documents["label"].tolist()
# Get the vocab, encoded dataset and document embeddings
deu = DocumentEmbeddingUtils(simplifiedDataFrame)
maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

x_train, x_test, y_train, y_test = train_test_split(encoded_documents, labels, test_size=0.2, shuffle=True)


def build_model(hp):
    input_layer = Input(name="InputOne", shape=(encoded_documents.shape[1],))

    embedding_layer = Embedding(
        len(unique_words_array),
        300,
        input_length=maximum_words,
        trainable=False,
        weights=[document_word_embeddings],
        name="EmbeddingLayer")(input_layer)

    model = GRU(units=hp.Choice(
        name="gru_unit_1",
        values=[64, 96, 128]
    ), recurrent_activation=tf.keras.activations.sigmoid, return_sequences=True,
    activation=tf.keras.activations.tanh,
    reset_after=True)(embedding_layer)

    model = Dropout(name="DropoutOne", rate=hp.Choice(
        name="dropout_layer_one",
        values=[0.3, 0.4, 0.5]
    ))(model)

    model = GRU(units=hp.Choice(
        "gru_unit_2",
        values=[24, 32, 48]
    ), recurrent_activation=tf.keras.activations.sigmoid,
        return_sequences=False,
    activation=tf.keras.activations.tanh,
    reset_after=True)(model)

    model = Dense(units=hp.Choice(
        name="dense_layer_one",
        values=[8, 16, 24]), name="DenseLayerOne")(model)

    model = Dropout(name="DropoutTwo", rate=hp.Choice(
        name="dropout_layer_two",
        values=[0.3, 0.4, 0.5]
    ))(model)

    output_layer = Dense(units=1, activation="sigmoid")(model)

    final_model = Model(inputs=[input_layer], outputs=[output_layer])

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice(
        name="learning_rate_param",
        values=[0.01, 0.05, 0.005]
    )), metrics=['accuracy',
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalseNegatives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Precision()],

    loss=tf.keras.losses.BinaryCrossentropy())

    final_model.summary()

    plot_model(final_model, to_file="../data/gru_rnn_experiment.png", show_shapes=True, show_layer_names=True)

    return final_model

model_training_results = kt.RandomSearch(build_model,
                                         objective='val_accuracy',
                                         directory='GRU_Experiment',
                                         project_name='gru_experiment',
                                         max_trials=3,
                                         executions_per_trial=2)

csv_logger = CSVLogger("../data/gru_rnn_results.csv", append=True, separator=';')

model_training_results.search(np.asarray(x_train), np.asarray(y_train),
                              validation_data=(np.asarray(x_test), np.asarray(y_test)),
                              epochs=20,
                              callbacks=[csv_logger])

best_params = model_training_results.get_best_hyperparameters(1)
bestModel = model_training_results.get_best_models(1)
model_training_results.results_summary()
print(best_params)
