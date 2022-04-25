import spellchecker as sc
import nltk.tokenize as tk
import pandas as pd
import nltk.corpus as cp
import nltk.stem as st
from StemmerEnums import StemmerEnums
import sklearn.preprocessing as pp
from sklearn.utils import shuffle
from textblob import TextBlob
import numpy as np


class PreProcessingUtils:
    def __init__(self, dataFrame, columnNames):
        self.dataFrame = dataFrame
        self.columnNames = columnNames
        if self.columnNames is None and self.dataFrame is None:
            raise Exception("You need to supply a data frame and columnNames")

    def write_to_file(self):
        columns = self.dataFrame.columns
        self.dataFrame.to_csv("data/processed_list.csv", header=0)
        print("Processed data written to file")

    def concatenate_title_and_body(self):
        """This function concatenates both title and body into a singular text. I hate pandas. Some level of
        preprocessing must have occurred here."""
        marks = ['"', '~', '@', '$', '%', '^', '&', '(', ')', '_', '+', '/', '*', ',', '.', '?', '{', '}', '|', '[',
                 ']', ';', ':', '\'', '`', '<', '>', '-', '=']

        finalDataSet = pd.DataFrame()
        finalDataSet["text"] = ""

        print("Now concatenating title and article contents into a singular cell")
        for i, row in self.dataFrame.iterrows():
            concatednatedData = str(str(row["title"]) + " " + str(row["text"]))
            finalSentence = ""
            for w in concatednatedData:
                if w not in marks:
                    finalSentence = finalSentence + finalSentence.join(w)

            finalDataSet = finalDataSet.append({"text" : finalSentence}, ignore_index=True)
            print("Article title and body of article " + str(i) + " have been concatenated")

        return finalDataSet

    def auto_correct_word_spelling(self):
        print("Performing automated spelling correction. ")

        rowNum = 0
        for i, row in self.dataFrame.iterrows():

            if not isinstance(row["text"], str):
                rowNum = rowNum + 1
                continue

            content = row["text"]
            tb = TextBlob(text= content)
            corrected = tb.correct()
            self.dataFrame.at[rowNum, "text"] = corrected
            rowNum = rowNum + 1

            print("Article " + str(i) + " has been checked for spelling")

    def remove_special_characters(self, specialCharacters):

        charArray = []
        #specialCharacters is a dataFrame(text file)
        symbols = specialCharacters[0].tolist()

        for i in symbols:
            charArray.append(i.strip())
        #there are two columns to be processed - title and body
        for column in self.columnNames:
            print("Now processing text in column (removing special characters): " + str(column))

            for index, row in self.dataFrame.iterrows():

                cleansedWords = []
                if not isinstance(row[column], str):
                    continue
                #tokenize text into word tokens
                wordTokens = tk.word_tokenize(row[column], language='english')
                #only keep words (tokens) that do not appear in the list of special characters
                cleansedWords = [w for w in wordTokens if w not in charArray]
                self.dataFrame.at[index, column] = ' '.join(cleansedWords)
                print("Article ID " + str(index) + " special character removal successfully processed and saved")

            print("Done removing special characters in text in column: "+ str(column))

    def remove_stop_words(self):
        """This function handles the removal stop words from the corpus. Stop words are words that do not add value to the body of text"""
        #:cp = ntlk.corpus alias
        #load the pre-populated english stopwords list (part of the nltk package)
        stopWords = cp.stopwords.words('english')
        #two columns are loaded into this class, the title and body columns
        for column in self.columnNames:

            row_number = 0
            #dataFrame is the loaded fake and real news dataset.
            for index, row in self.dataFrame.iterrows():
                if not isinstance(row[column], str):
                    row_number = row_number + 1
                    continue
                #split the text into word tokens
                tokenizedWords = tk.word_tokenize(row[column], language='english', preserve_line=False)
                #only keep words that do not appear in the stopwords list
                cleansedWords = [w for w in tokenizedWords if w not in stopWords]
                self.dataFrame.at[index, column] = ' '.join(cleansedWords)
                row_number = row_number + 1
                print("Done processing stop word removal of article id "+str(index))

        print("Stop word removal process now complete.")

    def stem_words(self, stemmer):
        print("Now starting process to stem words")
        selectedStemmer = None

        if stemmer == StemmerEnums.PORTER_STEMMER:
            selectedStemmer = st.PorterStemmer()
        elif stemmer == StemmerEnums.SNOWBALL_STEMMER:
            selectedStemmer = st.SnowballStemmer()
        else:
            selectedStemmer = st.LancasterStemmer()

        for column in self.columnNames:
            for index, row in self.dataFrame.iterrows():
                words = tk.word_tokenize(row[column])
                stemmedWords = [selectedStemmer.stem(w) for w in words]
                self.dataFrame.at[index, column] = ' '.join(stemmedWords)
                print("Done stemming on column " + column + " with article " + str(index))

        print("Word stemming process complete")
        print("")

    def shuffle_array(self):
        dataFrame = shuffle(self.dataFrame)
        dataFrame.reset_index(inplace=True, drop=True)
        self.dataFrame = dataFrame

    def get_data_frame(self):
        return self.dataFrame

    def get_columns(self):
        return self.columnNames

    def get_encoded_labels(possibleLabels):
        labelEncoder = pp.LabelEncoder()
        return labelEncoder.fit_transform(possibleLabels)


    def get_cleansed_dataframe(self):
        return self.dataFrame

    def get_reshaped_features(self, featuresArray):
        if featuresArray is None:
            raise Exception("Please supply features data")

        numFeatures = featuresArray.shape[1]
        numSamples = featuresArray.shape[0]
        timeSteps = 1
        print(np.shape(featuresArray))
        newArr = np.reshape(featuresArray, (numSamples, timeSteps, numFeatures))
        print("New array shape: " + str(np.shape(newArr)))
        return newArr

    def remove_blank_rows(self):
        print("Scanning for rows to drop")
        rowsToDrop = pd.DataFrame()

        for index, row in self.dataFrame.iterrows():
            if not isinstance(row["text"], str) or not isinstance(row["title"], str):
                rowsToDrop.append(row)
                continue

        numFakeNewsDropped = 0
        numRealNewsDropped = 0

        self.dataFrame.drop(rowsToDrop.index, axis=0)

        if len(rowsToDrop) > 0:
            for i in rowsToDrop:
                if i["label"] == "fake":
                    numFakeNewsDropped = numFakeNewsDropped + 1
                else:
                    numRealNewsDropped = numRealNewsDropped + 1

        print("Total Fake News Rows Dropped: " + str(numFakeNewsDropped))
        print("Total Real News Rows Dropped: " + str(numRealNewsDropped))

        return self.dataFrame