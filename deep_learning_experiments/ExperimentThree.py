import kerastuner as kt
import pandas as pd
import tensorflow.keras.layers as l
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.models as m
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model

from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

documents = pd.read_csv("../data/summarized_text_11.csv",
                        error_bad_lines=False,
                        sep=",")

special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame()
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
rawDocumentLabels = np.asarray(cleaned_documents["label"].tolist())
deu = DocumentEmbeddingUtils(cleaned_documents)

maximum_words = deu.get_document_length(trimmed_documents)
unique_words_array, encoded_documents = deu.get_unique_word_count(maximum_words, trimmed_documents)

encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

imp_x_train, imp_x_test, imp_y_train, imp_y_test = train_test_split(encoded_documents,
                                                                    rawDocumentLabels,
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    shuffle=True)

input_layer = l.Input(name="InputLayer", shape=(encoded_documents.shape[1],))

model = l.Embedding(input_dim=len(unique_words_array),
                        output_dim=300,
                        name="EmbeddingLayer",
                        trainable=False,
                        weights=[document_word_embeddings],
                        )(input_layer)

model = l.Conv1D(name="Conv1D", activation='relu', filters=70, strides=3, kernel_size=5)(model)

model = l.MaxPool1D(pool_size=4, name="MaxPool1D")(model)

model = l.Dropout(name="DropoutOne", rate=0.3)(model)

model = l.LSTM(name="LSTMOne",  units=32, activation='tanh', stateful=False, return_sequences=False,
               recurrent_activation='sigmoid')(model)

model = l.Dropout(rate=0.3, name="DropoutTwo")(model)

model = l.Dense(units=10, activation='relu', name="DenseOne")(model)

output_layer = l.Dense(units=1, activation='sigmoid', name="OutputLayer")(model)

compiled_model = m.Model(inputs=[input_layer], outputs=[output_layer], name="CNNandRNN")

compiled_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy',
                tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                 tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
                 tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()],
                                                          loss='binary_crossentropy')

compiled_model.summary()

csvFile = CSVLogger(filename="../data/experiment_cnn_dnn_stacked_1.csv", separator=";", append=True)
plot_model(compiled_model, to_file="../data/experiment_cnn_dnn_stacked_1.png", show_shapes=True, show_layer_names=True)
compiled_model.fit(x=imp_x_train, y=imp_y_train, callbacks=[csvFile], validation_data=(imp_x_test, imp_y_test),
             epochs=20, batch_size=64)