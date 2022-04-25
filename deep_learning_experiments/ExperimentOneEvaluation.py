import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Embedding, Conv1D, MaxPool1D, Dropout, Flatten, Dense, Add, Concatenate
from tensorflow.python.keras.models import Sequential, Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils
import kerastuner as kt
import datetime

"""Source: https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""Explicit Text Features: Fetch Explicit Features from File, create DNN for implicit features"""
explicitFeatures = pd.read_csv("../data/feature_set_july_20_2.csv")
labels = explicitFeatures["label"].tolist()
explicitFeatures.drop([explicitFeatures.columns[0], explicitFeatures.columns[1]], axis=1, inplace=True)
values = explicitFeatures.iloc[:, 1:]

explicitModelFeatures = np.asarray(explicitFeatures)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_explicit_features = scaler.fit_transform(explicitModelFeatures)

explicit_x_train, explicit_x_test, explicit_y_train, explicit_y_test = train_test_split(scaled_explicit_features,
                                                                                        labels,
                                                                                        test_size=0.2,
                                                                                        train_size=0.8,
                                                                                        shuffle=False)

one_hot_encoder = OneHotEncoder()
train_encoded_labels = to_categorical(explicit_y_train, 2)
test_encoded_labels = to_categorical(explicit_y_test, 2)

# explicitModel = l.Dropout(name="Dropout2", rate=0.3)(explicitModel)
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

# summarized = deu.generate_summarized_text()

maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

imp_x_train, imp_x_test, imp_y_train, imp_y_test = train_test_split(encoded_documents,
                                                                    labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2)

implicit_features = np.asarray(encoded_documents)


def build_hybrid_model():
    explicit_model_input = l.Input(name="InputLayer1", shape=(explicitModelFeatures.shape[1],))

    explicit_model = l.Dense(name="DenseLayer1", units=384, activation=tf.keras.activations.relu,
                             bias_initializer=tf.keras.initializers.Zeros(),
                             kernel_initializer=tf.keras.initializers.GlorotUniform())(explicit_model_input)

    explicit_model = l.Dropout(name="Dropout1", rate=0.5)(explicit_model)

    explicit_model = l.Dense(name="DenseLayer2", units=48, activation=tf.keras.activations.relu)(explicit_model)

    embedding_inputs = l.Input(name="InputLayer2", shape=(maximum_words,))
    embedding_layer = Embedding(len(unique_words_array), 300, trainable=False, weights=[document_word_embeddings],
                                input_length=len(unique_words_array),
                                name="EmbeddingOne")

    implicit_model = embedding_layer(embedding_inputs)
    implicit_model = Conv1D(name="Conv1DOne", filters=90, kernel_size=4, activation='relu', strides=4,
                            kernel_initializer=tf.keras.initializers.GlorotUniform(),
                            padding='same', bias_initializer=tf.keras.initializers.Zeros())(implicit_model)

    implicit_model = MaxPool1D(name="MaxPoolOne", pool_size=3)(implicit_model)

    implicit_model = Dropout(name="DropoutLayer3", rate=0.5)(implicit_model)

    implicit_model = Flatten()(implicit_model)

    merged = Concatenate(name="Merge")([explicit_model, implicit_model])
    final_layer = Dense(units=128, activation=tf.keras.activations.relu)(merged)
    final_layer = Dropout(rate=0.3)(final_layer)

    final_layer = Dense(units=1, activation=tf.keras.activations.sigmoid)(final_layer)
    """Merge both implicit and explicit inputs - one output for the predicted class"""
    final_model = Model(inputs=[embedding_inputs, explicit_model_input], outputs=[final_layer], name="CNN_DNN")
    final_model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy',
                                                                       tf.keras.metrics.TrueNegatives(),
                                                                       tf.keras.metrics.TruePositives(),
                                                                       tf.keras.metrics.FalseNegatives(),
                                                                       tf.keras.metrics.FalsePositives(),
                                                                       tf.keras.metrics.AUC(),
                                                                       tf.keras.metrics.Recall(),
                                                                       tf.keras.metrics.Precision(),
                                                                       ],
                        loss=tf.keras.losses.BinaryCrossentropy())
    plot_model(final_model, to_file="../data/cnn_dnn_experiment_final_model_eo.png", show_shapes=True,
               show_layer_names=True)
    final_model.summary()

    csv_file = CSVLogger(filename="../data/cnn_dnn_2_1_run_eo_epoch.csv", separator=';', append=False)

    tensor_board_model = tf.keras.callbacks.TensorBoard(log_dir="../data/cnn_dnn/log_"
                                                                +datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                        histogram_freq=1)

    #return final_model
    final_model.fit(x=[np.asarray(imp_x_train), np.asarray(explicit_x_train)],
                   y=np.asarray(imp_y_train), callbacks=[csv_file, tensor_board_model],
                  validation_data=([np.asarray(imp_x_test), np.asarray(explicit_x_test)], np.asarray(imp_y_test)),
                 epochs=20)


build_hybrid_model()

#keras_classifier = KerasClassifier(build_fn=build_hybrid_model, epochs=20)
#k_fold_validation = KFold(n_splits=10, shuffle=True)
#results = cross_val_score(keras_classifier, X=np.concatenate(imp_x_train, explicit_x_train, axis=-1),
 #                         y=np.concatenate(imp_y_train, explicit_y_train), cv=k_fold_validation)
