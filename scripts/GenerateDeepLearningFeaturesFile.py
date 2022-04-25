import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#Get handcraft features from one text file
from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

print("This file generates the feature set to save time")

explicitFeatures = pd.read_csv("../data/feature_set_july_20_2.csv")
explicitFeatures.drop([explicitFeatures.columns[0], explicitFeatures.columns[1]], axis=1, inplace=True)
explicitModelFeatures = np.asarray(explicitFeatures)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_explicit_features = scaler.fit_transform(explicitModelFeatures)

#Get document encodings, and document embeddings
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
maximum_words = deu.get_max_document_length()
unique_words_array, encoded_documents = deu.get_unique_words(maximum_words)
encoded_documents = deu.pad_corpus_words(max_word_count=maximum_words, encoded_dataset=encoded_documents)
document_word_embeddings = deu.get_document_embeddings(300, maximum_words, unique_words_array)

finalCsv = pd.concat([pd.DataFrame(encoded_documents, index=None), explicitFeatures], ignore_index=True)
finalCsv.to_csv("../data/deep_learning_features_file.csv", header=True)

unique_words_found = pd.DataFrame(unique_words_array)
unique_words_found.to_csv("../data/unique_word_list.csv", header=False)

print("Done concatenating feature set and unique words array!")