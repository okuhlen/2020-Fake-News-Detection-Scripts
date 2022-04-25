import pandas as pd
import matplotlib as plt
import nltk.tokenize as tk
from nltk.corpus import stopwords
from nltk.stem import snowball
import re as r
import sklearn.feature_extraction.text as sl
import sklearn.utils as ut
import sklearn.model_selection as ms
import sklearn.svm as s
import sklearn.metrics as pm
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.preprocessing as l

def pre_process_data(dataFrame):

    stopWords = stopwords.words('english')
    stemmer = snowball.SnowballStemmer('english', ignore_stopwords=True)
    for i, row in dataFrame.iterrows():
        content_without_stopwords = []
        title_without_stopwords = []
        title = r.sub('[^A-Za-z]', ' ', row["title"])
        content = r.sub('[^A-Za-z]', ' ', row["text"])
        title = title.lower()
        content = content.lower()
        title = stemmer.stem(title)
        content = stemmer.stem(content)
        tokenizedTitle = tk.word_tokenize(title, 'english')
        tokenizedContent = tk.word_tokenize(content, 'english')
        title_without_stopwords = [w for w in tokenizedTitle if w not in stopWords]
        content_without_stopwords = [w for w in tokenizedContent if w not in stopWords]
        dataFrame[i, "title"] = ' '.join(title_without_stopwords)
        dataFrame[i, "text"] = ' '.join(content_without_stopwords)

    return dataFrame

def get_bag_of_words_model(dataFrame, columnName):

    bowMatrix = sl.CountVectorizer(max_features=1000)
    columnArray = dataFrame[columnName].tolist()
    X = bowMatrix.fit_transform(columnArray)
    print(X.toarray())
    return X

def train_model(feature_vectors, y_labels):

    X_Train, X_Test, Y_Train, Y_Test = ms.train_test_split(feature_vectors, y_labels, train_size=0.7, test_size=0.3)
    svmClassifier = s.LinearSVC()
    svmClassifier.fit(X_Train, Y_Train)
    predictedLabels = svmClassifier.predict(X_Test)
    accuracy = pm.accuracy_score(Y_Test, predictedLabels)
    auc = pm.auc(X_Train, Y_Train)
    print('Accuracy: ' + str(accuracy))

    #TODO: Calculate the confustion matrix

    #chart = plt.scatter(x=X_Train[0:, 0], y=Y_Train[0:, 1], cmap=plt.cm.colors.LinearSegmentedColormap, c=(0, 1))
    #chart.xlabel('Articles')
    #chart.ylabel('Fake/Real News')
    #chart.title('Sample Classification test')
    #plt.show()


##Begin of the script

fakeNewsDataSource = pd.read_csv("C:/Projects/Research Projects/Master of Information Technology/Datasets/Fake.csv",
                                  error_bad_lines=False, nrows=1500)
realNewsDataSource = pd.read_csv("C:/Projects/Research Projects/Master of Information Technology/Datasets/True.csv",
                                  error_bad_lines=False, nrows=1500)

print("Adding fake/real labels to the dataset")
fakeNewsDataSource["label"] = "fake"
realNewsDataSource["label"] = "real"
print("Merging fake and real news datasets...")
mergedData = pd.concat([fakeNewsDataSource, realNewsDataSource])
newDataSource = pre_process_data(mergedData)
print("Shuffling the data...")
newDataSource.reset_index(drop=True)
newDataSource = ut.shuffle(newDataSource)
contentBagOfWordsModel = get_bag_of_words_model(newDataSource, 'text')

print("Transforming labels...")
labelTransformer = l.LabelEncoder()

labels = newDataSource['label'].tolist()
transformedLabels = labelTransformer.fit_transform(labels)
print(transformedLabels)

print(newDataSource.head())
print("Now performing svm classification on BoW vectors")

train_model(contentBagOfWordsModel, transformedLabels)
print("End of Script")
