import tensorflow as tf
import tensorflow.keras.layers as l
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Embedding, Conv1D, MaxPool1D, Dropout, Flatten, Dense, Add, Concatenate
from tensorflow.python.keras.models import Sequential, Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from PreProcessingUtils import PreProcessingUtils
from data_visualizations.DataVisualizationUtility import DataVisualizationUtility
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

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
"""DEFINE DNN MODEL FOR EXPLICIT TEXT FEATURES"""
explicitModelInput = l.Input(name="InputLayer1", shape=(explicitModelFeatures.shape[1],))
explicitModel = l.Dense(name="DenseLayer1", units=192, activation=tf.keras.activations.relu,
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(explicitModelInput)
explicitModel = l.Dropout(name="Dropout1", rate=0.5)(explicitModel)
explicitModel = l.Dense(name="DenseLayer2", units=64, activation=tf.keras.activations.relu)(explicitModel)
# explicitModel = l.Dropout(name="Dropout2", rate=0.3)(explicitModel)
"""Implicit Text Features: Basic pre-processing, create CNN model."""
documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv", chunksize=5000, error_bad_lines=False, sep=",",
                        nrows=6000)
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

document_visualisation = DataVisualizationUtility(cleaned_documents)
document_visualisation.generate_word_cloud()
summarized = deu.generate_summarized_text()

maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

imp_x_train, imp_x_test, imp_y_train, imp_y_test = train_test_split(encoded_documents,
                                                                    labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2)

implicit_features = np.asarray(encoded_documents)

# Define input layer
embedding_inputs = l.Input(name="InputLayer2", shape=(maximum_words,))
# Define embedding layer - use word embeddings as initial weights
embedding_layer = Embedding(len(unique_words_array), 300, trainable=False, weights=[document_word_embeddings],
                            input_length=len(unique_words_array),
                            name="EmbeddingOne")

implicitModel = embedding_layer(embedding_inputs)
# Declare and configure Conv1D cell
implicitModel = Conv1D(name="Conv1DOne", filters=5, kernel_size=3, activation='relu',
                       strides=2, kernel_initializer=tf.keras.initializers.GlorotUniform(), padding='same',
                       bias_initializer=tf.keras.initializers.Zeros())(implicitModel)
# Add Max-Pooling layer
implicitModel = MaxPool1D(name="MaxPoolOne", pool_size=3)(implicitModel)
# Regularization technique: Add dropout layer
implicitModel = Dropout(name="DropoutLayer3", rate=0.5)(implicitModel)
implicitModel = Flatten()(implicitModel)

# Merge the two models - CNN and the DNN
merged = Concatenate(name="Merge")([explicitModel, implicitModel])

# add final layers
finalLayer = Dense(units=128, activation=tf.keras.activations.relu)(merged)
finalLayer = Dropout(rate=0.4)(finalLayer)

# output layer - use sigmoid function for binary classification problems
finalLayer = Dense(units=1, activation=tf.keras.activations.sigmoid)(finalLayer)
"""Merge both implicit and explicit inputs - one output for the predicted class"""
finalModel = Model(inputs=[embedding_inputs, explicitModelInput], outputs=[finalLayer], name="CNN_DNN")
# OPTIONAL: Log training results to CSV file for further analysis
csvFile = CSVLogger(filename="../data/cnn_dnn_2_1.csv", separator=';', append=False)

# Compile the model
finalModel.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy',
                                                                  tf.keras.metrics.TrueNegatives(),
                                                                  tf.keras.metrics.TruePositives(),
                                                                  tf.keras.metrics.FalseNegatives(),
                                                                  tf.keras.metrics.FalsePositives(),
                                                                  tf.keras.metrics.AUC(),
                                                                  tf.keras.metrics.Recall(),
                                                                  tf.keras.metrics.Precision(),
                                                                  ],
                   loss=tf.keras.losses.BinaryCrossentropy())
plot_model(finalModel, to_file="../data/merged_model_final13_2.png", show_shapes=True, show_layer_names=True)
finalModel.summary()
finalModel.fit(x=[np.asarray(imp_x_train), np.asarray(explicit_x_train)],
               y=np.asarray(imp_y_train), callbacks=[csvFile],
               validation_data=([np.asarray(imp_x_test), np.asarray(explicit_x_test)], np.asarray(imp_y_test)),
               epochs=20)
