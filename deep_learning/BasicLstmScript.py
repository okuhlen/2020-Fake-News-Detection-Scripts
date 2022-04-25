from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
import pandas as pd
from tensorflow.python.keras.layers import Embedding, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from FeatureUtils import FeatureUtils
from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

examplerDataSet = pd.read_csv("../data/feature_set_with_doc_vectors_july_final.csv")


mergedDataset = pd.read_csv("../data/merged_fake_real_dataset.csv", chunksize=5000, error_bad_lines=False,
                                                                                                          sep=",")
specialCharacters = pd.read_csv("../data/special_characters.txt", sep='-', header=None, low_memory=True)
featureDataSet = pd.DataFrame()
cleanedFrame = pd.DataFrame()

for i in mergedDataset:

    pp = PreProcessingUtils(i, ["title", "text"])
    pp.remove_stop_words()
    pp.remove_special_characters(specialCharacters)
    #pp.auto_correct_word_spelling()
    cleanedFrame = pd.concat([cleanedFrame, pp.get_data_frame()])

featureDataSet["label"] = cleanedFrame["label"]
fu = FeatureUtils(cleanedFrame, ["title", "text"])

docUtils = DocumentEmbeddingUtils(cleanedFrame)
word_vocab, vocab_list = docUtils.get_document_vocab_size()

#define the model
model = Sequential()
model.add(Embedding(len(vocab_list), featureDataSet.shape[0], input_length=featureDataSet.shape[1]))
model.add(LSTM(activation='relu', units=128))
model.add(Dropout())
