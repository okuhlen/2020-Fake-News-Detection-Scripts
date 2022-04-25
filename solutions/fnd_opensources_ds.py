import pandas as pd
import numpy as np
import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sklearn as sc
import csv as c
import sys as s
import time as t
import os as o
import bs4 as bs
import seaborn as sb
#remove stopwords at column 5 [content] and 9 for [title]
def removeStopWordsFromText(dataFrame, column):

    dataFrame.reset_index(inplace=True, drop=True)
    stemmer = nl.stem.SnowballStemmer("english", ignore_stopwords=True)

    for index, row in dataFrame.iterrows():

        rawText = row.at[column]

        word_tokens = word_tokenize(rawText, language="english")
        cleaned_words = [w for w in word_tokens if not w in stopwords.words()]

        stemPointer = 0
        if (cleaned_words is not None):
            for w in cleaned_words:
                cleaned_words[stemPointer] = stemmer.stem(w)
                stemPointer += 1

        dataFrame.at[index, column] = " ".join(cleaned_words)

    print("Done Processing!")
    return dataFrame

def getTf_IdfTransformer(word_count_vector, countVectorizer):

    vectorizer = sc.feature_extraction.text.TfidfTransformer(smooth_idf=True, use_idf=True)
    X = vectorizer.fit(word_count_vector)

    idfDataFrame = pd.DataFrame(vectorizer.idf_, countVectorizer.get_feature_names())

    print("")
    return

def getTf_IdfVecotizer(dataFrame):

    tfVectorizer = sc.feature_extraction.text.TfidfVectorizer()
    X = tfVectorizer.fit_transform(dataFrame["content"].tolist())
    cols = tfVectorizer.get_feature_names()
    df = pd.DataFrame(data=X.toarray(), columns=cols)

    print(X)

totalDocuments = 0



c.field_size_limit(2147483647)
openSourcesList = []

for chunk in pd.read_csv("C:/Projects/Research Projects/Master of Information Technology/Datasets/OpenSources.csv",
                         sep=",", header=0, chunksize=1000, iterator=True, error_bad_lines=False,
                        skiprows=0, skip_blank_lines=True, engine='python', nrows=5):
    openSourcesList.append(chunk)

openSourcesDataFrame = pd.concat(openSourcesList)

print("Now sanitizing text from the body")
openSourcesDataFrame = removeStopWordsFromText(openSourcesDataFrame, "content")
print("Now sanitizing stopwords from the tile")
openSourcesDataFrame = removeStopWordsFromText(openSourcesDataFrame, "title")

##get TF features from the body column
wordCountVectorizer = sc.feature_extraction.text.CountVectorizer()
documents = openSourcesDataFrame["content"].tolist()
wordCountVectors = wordCountVectorizer.fit_transform(documents)
print(wordCountVectors.toarray())

getTf_IdfVecotizer(openSourcesDataFrame)
#getTf_IdfTransformer(wordCountVectors, wordCountVectorizer)









