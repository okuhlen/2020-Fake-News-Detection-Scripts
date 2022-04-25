import pandas as pd
import gensim as g
from FeatureUtils import FeatureUtils
from PreProcessingUtils import PreProcessingUtils

featureDataSet = pd.read_csv("../data/feature_set_july_20_2.csv", low_memory=False, sep=",")
doc2vecModel = g.models.doc2vec.Doc2Vec.load("../data/opensources_full_ds.model")
mergedDataset = pd.read_csv("../data/merged_fake_real_dataset.csv", chunksize=6000, error_bad_lines=False, sep=",")
specialCharacters = pd.read_csv("../data/special_characters.txt", sep='-', header=None, low_memory=True)

cleanedFrame = pd.DataFrame()

print("Performing pre-processing of text")

for i in mergedDataset:
    pp = PreProcessingUtils(i, ["title", "text"])
    pp.remove_stop_words()
    pp.remove_special_characters(specialCharacters)
    pp.auto_correct_word_spelling()
    cleanedFrame = pd.concat([cleanedFrame, pp.get_data_frame()])

print("Now appending features to feature-set")

cleanedFrame.reset_index()
fu = FeatureUtils(cleanedFrame, ["title", "text"])

featureDataSet = fu.get_word_count(featureDataSet)
featureDataSet = fu.calculate_ttr_and_unique_words(featureDataSet)
featureDataSet = fu.get_average_sentence_length(featureDataSet, "text")
featureDataSet = fu.get_readability_score(featureDataSet)
featureDataSet = fu.calculate_cosine_similarity(featureDataSet)

print("Now saving new feature set to file")
featureDataSet.to_csv("../data/feature_set_july_2020.csv", header=True)
print("Done!")


