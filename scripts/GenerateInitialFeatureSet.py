import pandas as pd
from FeatureUtils import FeatureUtils
from PreProcessingUtils import PreProcessingUtils
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

featureDataSet = fu.calculate_ttr_and_unique_words(featureDataSet)
featureDataSet = fu.calculate_cosine_similarity(featureDataSet)
featureDataSet = fu.get_readability_score(featureDataSet)
featureDataSet = fu.get_average_sentence_length(featureDataSet, "text")
#featureDataSet = fu.get_punctuation_mark_count(featureDataSet)
featureDataSet = fu.get_word_count(featureDataSet)
featureDataSet = fu.get_parts_of_speech_tags(featureDataSet, "text")

featureDataSet.to_csv("../data/feature_set_july_20_2.csv", header=True)
print("Done!")