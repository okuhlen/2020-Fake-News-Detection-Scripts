import pandas as pd
import nltk.tokenize as tk
from string import punctuation
import lexicalrichness as lr
from gensim.models import Doc2Vec
from readability import Readability
from nltk import pos_tag
import math as m
from scipy import spatial
import gensim as g
import math as m
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import sklearn.preprocessing as pp


class FeatureUtils:
    def __init__(self, dataFrame, columnNames):
        if dataFrame is None or columnNames is None:
            raise Exception("The data source and columns are missing")
        self.dataFrame = dataFrame
        self.columnNames = columnNames

    def create_feature_datafame(self):
        return pd.DataFrame()

    def get_word_count(self, featureDataFrame):
        counter = 0
        print("Calculating word count for the selected columns")
        featureDataFrame["title_wordCount"] = 0
        featureDataFrame["text_wordCount"] = 0
        for col in self.columnNames:

            for index, row in self.dataFrame.iterrows():

                if not isinstance(row[col], str):
                    featureDataFrame.at[index, col + "_wordCount"] = 0
                    counter = counter + 1
                    continue

                wordTokens = tk.word_tokenize(row[col], language='english')
                featureDataFrame.at[index, col + "_wordCount"] = len(wordTokens)
                print("Article " + str(counter) + "has a word count of : " +str(len(wordTokens)))
                counter = counter + 1

            print("Column " + col + " word count processed successfully")
            counter = 0
        print("Calculation of word count for selected columns complete")
        return featureDataFrame

    #pos tager does this. not needed.
    def get_punctuation_mark_count(self, featureDataFrame):
        punctuationMarks = punctuation
        countMarks = 0
        print("Now calculating punctuation mark count")
        for col in self.columnNames:
            featureDataFrame[col + "_punctuationCount"] = []
            for index, row in self.dataFrame.iterrows():
                content = row[col]
                for i in content:
                    if i in punctuationMarks:
                        countMarks = countMarks + 1
                featureDataFrame[index, col + "_punctuationCount"] = countMarks
        print("Punctuation mark count calculated successfully")
        print()
        return featureDataFrame

    def get_readability_score(self, featureDataFrame):
        print("Now calculating readability score for articles")
        ari =  "text_ari"
        gfIndex = "text_gf"
        fkIndex = "text_fkg"
        featureDataFrame[ari] = float(0.0) # automated readability index
        featureDataFrame[gfIndex] = float(0.0) # fog index
        featureDataFrame[fkIndex] = float(0.) # Flesch Kincaid Grade Level index

        counter = 0

        for index, row in self.dataFrame.iterrows():

            if not isinstance(row["text"], str):
                featureDataFrame.at[counter, ari] = 0
                featureDataFrame.at[counter, gfIndex] = 0
                featureDataFrame.at[counter, fkIndex] = 0
                print(str(index))
                counter = counter + 1
                continue

            readabilityMetrics = Readability(row["text"])
            tokens = tk.word_tokenize(row["text"])

            if len(tokens) <= 140:
                featureDataFrame.at[counter, ari] = 0
                featureDataFrame.at[counter, gfIndex] = 0
                featureDataFrame.at[counter, fkIndex] = 0
                print(str(index))
                counter = counter + 1
                continue

            featureDataFrame.at[counter, ari] = readabilityMetrics.ari().score
            featureDataFrame.at[counter, gfIndex] = readabilityMetrics.gunning_fog().score
            featureDataFrame.at[counter, fkIndex] = readabilityMetrics.flesch_kincaid().score
            print("Done calculating readability metrics for article " + str(counter))
            counter = counter + 1


        print("Readability metrics as features successfully added to the dataset")
        print()
        return featureDataFrame

    def get_parts_of_speech_tags(self, featureDataFrame, columnName):

        print("Now generating POS tags")
        counter = 0

        for index, row in self.dataFrame.iterrows():

            if not isinstance(row[columnName], str):
                counter = counter + 1
                continue

            wordTokens = tk.word_tokenize(row[columnName], language="english")
            partsOfSpeech = pos_tag(wordTokens, lang="eng")

            for wordTag in partsOfSpeech:
                if wordTag[1] in featureDataFrame.columns:
                    tag = wordTag[1]
                    featureDataFrame.at[counter, tag] = featureDataFrame.at[counter, tag] + 1
                else:
                    pos = wordTag[1]
                    featureDataFrame[pos] = 0
                    featureDataFrame.at[counter, pos] = 1

            print("POS Tags successfully generated for article " + str(counter))
            counter = counter + 1

        print("Parts of speech features added to feature matrix. ")
        print(featureDataFrame.head())
        return featureDataFrame

    def get_average_sentence_length(self, featureDataFrame, columName):

        counter = 0
        print("Calculating average sentence length for content")
        for index, row in self.dataFrame.iterrows():
            countOnWords = 0

            if not isinstance(row[columName], str):
                featureDataFrame.at[counter, columName + "_avgSentLen"] = 0
                counter = counter + 1
                print("No content found for article id" + str(counter) + ". length is 0")
                continue

            sentenceTokens = tk.sent_tokenize(row[columName], language='english')

            if sentenceTokens is None or len(sentenceTokens) == 0:
                featureDataFrame.at[counter, columName + "_avgSentLen"] = 0
                counter = counter + 1
                print("No content found for article id" + str(counter)+ ". length is 0")
                continue

            for i in sentenceTokens:
                wordTokens = tk.word_tokenize(i, language='english')
                countOnWords = countOnWords + len(wordTokens)

            averageSentLengh = m.ceil(countOnWords / len(sentenceTokens))
            featureDataFrame.at[counter, columName+"_avgSentLen"] = averageSentLengh
            print("Average word length for article " + str(counter) + " is " + str(averageSentLengh))
            counter = counter + 1
            print("Calculation of average sentence length now complete")

        return featureDataFrame

    def calculate_cosine_similarity(self, featureDataFrame):

        scores_per_sentence = []
        doc2vecModel = Doc2Vec.load("../data/opensources_full_ds.model")
        aggregatedCosine = 0
        featureDataFrame["cosine_sim"] = float(0.0)
        rowNumber = 0

        for i, row in self.dataFrame.iterrows():

            if not isinstance(row["title"], str) or not isinstance(row["text"], str):
                featureDataFrame.at[rowNumber, "cosine_sim"] = 0
                rowNumber = rowNumber + 1
                continue

            titleVectors = doc2vecModel.infer_vector(row["title"].split())
            transformedTitleVectors = pp.minmax_scale(titleVectors, feature_range=(0, 1), axis=0, copy=True)
            sentences = tk.sent_tokenize(row["text"])

            print("Title vectors for article " + str(rowNumber))
            print(titleVectors)

            if sentences is not None:
                for sentence in sentences:
                    vectors = doc2vecModel.infer_vector(sentence.split())


                    #sentenceScore = spatial.distance.cosine(titleVectors, vectors)

                    transformedVectors = pp.minmax_scale(vectors, feature_range=(0,1), axis=0, copy=True)
                    newSentScore = metrics.pairwise.cosine_similarity(np.reshape(transformedTitleVectors, (1, -1)),
                                                                                 np.reshape(transformedVectors, (1,
                                                                                                                 -1)))
                    scores_per_sentence.append(newSentScore)

                if scores_per_sentence is None:
                    aggregatedCosine = 0
                else:
                    aggregatedCosine = np.mean(scores_per_sentence)

            print("Cosine similarity scores for Article " + str(rowNumber))
            print(scores_per_sentence)

            featureDataFrame.at[rowNumber, "cosine_sim"] = pd.to_numeric(round(float(aggregatedCosine), 2))
            scores_per_sentence = []
            print("Done calculating cosine similarity for Article " + str(rowNumber))
            rowNumber = rowNumber + 1

        return featureDataFrame

    def add_labels_to_vector(self, featureDataFrame):

        rowNum = 0
        print("Transforming labels")
        for i, row in self.dataFrame.iterrows():
            if row["label"] == "fake":
                self.dataFrame[rowNum, "label"] = 0
            else: #real news
                self.dataFrame[rowNum, "label"] = 1
            rowNum = rowNum + 1

        print("Copying labels")
        featureDataFrame["label"] = self.dataFrame["label"]
        print("Done transforming labels")
        return featureDataFrame

    """This function calculates the Type-Token Ratio (TTR) and the Unique Word Count in a document body"""
    def calculate_ttr_and_unique_words(self, featureDataFrame):
        print("Calculating Type-Token Ratio and Unique Words: ")
        featureDataFrame["tt_ratio"] = float(0.0)
        featureDataFrame["uq_words"] = 0
        rowCounter = 0

        for i, row in self.dataFrame.iterrows():
            content = row["text"]
            if not isinstance(content, str):
                print("TTR Ratio for Article " + str(rowCounter) + " is: 0 [NaT]")
                featureDataFrame.at[rowCounter, "tt_ratio"] = float(0.0)
                featureDataFrame.at[rowCounter, "uq_words"] = 0
                rowCounter = rowCounter + 1
                continue
            lex = lr.LexicalRichness(use_TextBlob=True, text=content)
            ttr_score = round(float(lex.ttr), 2)
            featureDataFrame.at[rowCounter, "tt_ratio"] = str(float(ttr_score))
            featureDataFrame.at[rowCounter, "uq_words"] = pd.to_numeric(lex.terms)
            rowCounter = rowCounter + 1
            print("TTR Ratio for Article " + str(rowCounter) + " is: " + str(ttr_score))
            print("Unique Words for Article " +str(rowCounter) + " is: " + str(lex.terms))

        return featureDataFrame



