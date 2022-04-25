import pandas as pd
from time import time
import PreProcessingUtils as p
import StemmerEnums as s
import FeatureUtils as f

print("Fake News detection script run one")
startTimeStamp = time()

mergedDataFrame = pd.read_csv("C:/Projects/Research Projects/Master of Information "
                              "Technology/Datasets/merged_dataset.csv", error_bad_lines=False, sep=',')

columnNames = ["title", "content"]
labels = ["fake", "satire", "bias", "conspiracy", "hate", "clickbait", "unreliable", "reliable"]
print("Now starting pre-processing functionality")
extractionHelper = p.PreProcessingUtils(dataFrame=mergedDataFrame, columnNames=columnNames)
extractionHelper.remove_stop_words()
extractionHelper.correct_word_spelling()
extractionHelper.stem_words(s.StemmerEnums.SNOWBALL_STEMMER)
mergedDataFrame = extractionHelper.get_cleansed_dataframe()
encodedLabels = extractionHelper.get_encoded_labels(possibleLabels=labels)
print(mergedDataFrame.head())
end = time()
diff = end - startTimeStamp

print("Pre processing now complete")
featureUtils = f.FeatureUtils(mergedDataFrame, columnNames)
featuresDataFrame = pd.DataFrame()
featuresDataFrame = featureUtils.get_word_count(featuresDataFrame)
print(featuresDataFrame.head())



