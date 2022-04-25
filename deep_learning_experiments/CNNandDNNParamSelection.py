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
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils
import kerastuner as kt

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

#summarized = deu.generate_summarized_text()

maximum_words = deu.get_max_document_length(False)
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

imp_x_train, imp_x_test, imp_y_train, imp_y_test = train_test_split(encoded_documents,
                                                                    labels,
                                                                    train_size=0.8,
                                                                    test_size=0.2)

implicit_features = np.asarray(encoded_documents)


def build_hybrid_model(hp):
    explicit_model_input = l.Input(name="InputLayer1", shape=(explicitModelFeatures.shape[1],))

    explicitModel = l.Dense(name="DenseLayer1", units=hp.Choice(
        name="explicit_dense_1",
        values=[160, 192, 256, 384]
    ), activation=tf.keras.activations.relu,
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.keras.initializers.GlorotUniform())(explicit_model_input)
    explicitModel = l.Dropout(name="Dropout1", rate=hp.Choice(
        name="explicit_dropout_1_units",
        values=[0.2, 0.3, 0.4, 0.5]
    ))(explicitModel)
    explicitModel = l.Dense(name="DenseLayer2", units=hp.Choice(
        name="explicit_dense_2",
        values=[32, 48, 64, 96]
    ), activation=tf.keras.activations.relu)(explicitModel)

    embedding_inputs = l.Input(name="InputLayer2", shape=(maximum_words,))
    embedding_layer = Embedding(len(unique_words_array), 300, trainable=False, weights=[document_word_embeddings],
                                input_length=len(unique_words_array),
                                name="EmbeddingOne")

    implicitModel = embedding_layer(embedding_inputs)
    implicitModel = Conv1D(name="Conv1DOne", filters=hp.Choice(
        name="conv1d_1_filters",
        values=[50, 70, 90, 120]
    ), kernel_size=hp.Choice(
        name="conv1d_1_kernels",
        values=[3, 4, 5]
    ), activation='relu',
        strides=hp.Choice(
            name="conv1d_1_strides",
            values=[2, 3, 4]
        ), kernel_initializer=tf.keras.initializers.GlorotUniform(),
        padding='same', bias_initializer=tf.keras.initializers.Zeros())(implicitModel)
    implicitModel = MaxPool1D(name="MaxPoolOne", pool_size=hp.Choice(
        name="pool_size",
        values=[3, 4, 5]
    ))(implicitModel)
    implicitModel = Dropout(name="DropoutLayer3", rate=hp.Choice(
        name="dropout_3",
        values=[0.3, 0.4, 0.5]
    ))(implicitModel)
    implicitModel = Flatten()(implicitModel)

    merged = Concatenate(name="Merge")([explicitModel, implicitModel])
    finalLayer = Dense(units=hp.Choice(
        name="final_layer_units",
        values=[128, 160, 192]
    ), activation=tf.keras.activations.relu)(merged)
    finalLayer = Dropout(rate=hp.Choice(
        name="final_dropout_layer",
        values=[0.3, 0.4, 0.5]
    ))(finalLayer)

    finalLayer = Dense(units=1, activation=tf.keras.activations.sigmoid)(finalLayer)
    """Merge both implicit and explicit inputs - one output for the predicted class"""
    finalModel = Model(inputs=[embedding_inputs, explicit_model_input], outputs=[finalLayer], name="CNN_DNN")
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
    plot_model(finalModel, to_file="../data/cnn_dnn_experiment_model_1.png", show_shapes=True, show_layer_names=True)
    finalModel.summary()

    return finalModel


csvFile = CSVLogger(filename="../data/cnn_dnn_experiment_results_data_1.csv", separator=';', append=True)
tuner = kt.RandomSearch(build_hybrid_model,
                        objective='val_accuracy',
                        directory='CNN_DNN_RES_1',
                        project_name='cnn_dnn_results_1',
                                      max_trials=3,
                                      executions_per_trial=2)

tuner.search(x=[np.asarray(imp_x_train), np.asarray(explicit_x_train)],
                   y=np.asarray(imp_y_train), callbacks=[csvFile],
                   validation_data=([np.asarray(imp_x_test), np.asarray(explicit_x_test)], np.asarray(imp_y_test)),
                   epochs=20)

best_params = tuner.get_best_hyperparameters(1)
bestModel = tuner.get_best_models(1)
tuner.results_summary()