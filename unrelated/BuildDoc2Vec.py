from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from nltk import word_tokenize

from PreProcessingUtils import PreProcessingUtils
import gensim as g
import numpy as np

print("Using open sources data sets")
loadedFiles = pd.DataFrame()

openSourcesSet = pd.read_csv("C:\Projects\Research Projects\Master of Information "
                             "Technology\Datasets\custom_opensources_dataset.csv",
                             chunksize=10000, skip_blank_lines=True,
                             iterator=True, sep=",", header=0, error_bad_lines=False, low_memory=False,
                             lineterminator="\n")

specialCharacters = pd.read_csv("../data/special_characters.txt", sep='-', header=None, low_memory=True)
print("Loading data")

for chunk in openSourcesSet:

    preProcessing = PreProcessingUtils(columnNames=["Content"], dataFrame=chunk)
    preProcessing.shuffle_array()
    preProcessing.remove_stop_words()
    preProcessing.remove_special_characters(specialCharacters)
    loadedFiles = pd.concat([loadedFiles, preProcessing.get_data_frame()])
    print(str(loadedFiles.shape[0]) + " articles cleansed and loaded to the dataset.")

print("Now attempting to build doc2vec model")

taggedDocuments = []
taggedCount = 0
for index, row in loadedFiles.iterrows():
    documentTokens = word_tokenize(row["Content"], language="english")
    taggedDocuments.append(TaggedDocument(documentTokens, [index]))
    print("Article successfully tagged. Article ID: " + str(taggedCount))
    taggedCount = taggedCount + 1

print("Constructing model")
doc2vecModel = Doc2Vec(vector_size=400, dm=1, max_vocab_size=None, documents=taggedDocuments,
                       workers=4, window=8)
doc2vecModel.save("../data/opensources_full_ds.model")

doc2vecModel.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
print("Doc2vec model successfully trained")

