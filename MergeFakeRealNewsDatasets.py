import pandas as pd
import PreProcessingUtils as pp
from data_visualizations.DataVisualizationUtility import DataVisualizationUtility

fakeNewsSet = pd.DataFrame()
realNewsSet = pd.DataFrame()
fakeNewsDataSet = pd.read_csv("C:/Datasets/Fake.csv", sep=",", error_bad_lines=False, engine='c', low_memory=True,
                              chunksize=1500, header=0, usecols=["title", "text"])
realNewsDataSet = pd.read_csv("C:/Datasets/True.csv", error_bad_lines=False, sep=",", engine='c', low_memory=True,
                              chunksize=1500, header=0, usecols=["title", "text"])
specialCharacters = pd.read_csv("data/special_characters.txt", sep='-', header=None, low_memory=True)
print("Now cleaning data")
headlineAndBody = ["title", "text"]

print("Now processing Fake News Articles")
for i in fakeNewsDataSet:
    i["label"] = 1 #fake
    preProcessor = pp.PreProcessingUtils(dataFrame=i, columnNames=headlineAndBody)
    preProcessor.shuffle_array()
    preProcessor.remove_stop_words()
    #preProcessor.stem_words(StemmerEnums.StemmerEnums.PORTER_STEMMER)
    preProcessor.remove_special_characters(specialCharacters)
    fakeNewsSet = pd.concat([fakeNewsSet, preProcessor.get_data_frame()])

print("Now processing real news articles")
for i in realNewsDataSet:
    i["label"] = 0 #real
    preProcessor = pp.PreProcessingUtils(dataFrame=i, columnNames=headlineAndBody)
    preProcessor.shuffle_array()
    preProcessor.remove_stop_words()
    #preProcessor.stem_words(StemmerEnums.StemmerEnums.PORTER_STEMMER)
    preProcessor.remove_special_characters(specialCharacters)
    realNewsSet = pd.concat([realNewsSet, preProcessor.get_data_frame()])

print("Mering all data sources...")
mergedDataset = pd.concat([fakeNewsSet, realNewsSet])

print("Final dataset contains "+ str(mergedDataset.shape[0]) +" rows.")
print("Fake news dataset contains "+str(fakeNewsSet.shape[0]) + " rows.")
print("Real news dataset contains "+str(realNewsSet.shape[0]) + " rows.")
print("")

preProcessor = pp.PreProcessingUtils(dataFrame=mergedDataset, columnNames=headlineAndBody)
mergedSet = preProcessor.remove_blank_rows()

mergedSet.to_csv("data/merged_fake_real_dataset_sept.csv", header=True)
print("Now generating word cloud for both fake and real news data sets")

dataVisionUtils = DataVisualizationUtility(mergedDataset)

dataVisionUtils.generate_word_cloud()
print("Done")




