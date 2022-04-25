import datetime as d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from CrossValidationUtils import CrossValidationUtils
from graphing.GraphHelper import GraphHelper
from scripts.HyperParameterSelectionUtils import HyperParameterSelectionUtils
from scripts.ParamSets import ParamSets

print("Starting script")
featureDataSet = pd.read_csv("../data/feature_set_with_doc_vectors_july_final.csv", low_memory=False, sep=",")
print("Removing un-used index columns")

timeStart = d.datetime.now()
dt_string = timeStart.strftime("%d/%m/%Y %H:%M:%S")

featureDataSet.drop([featureDataSet.columns[0], featureDataSet.columns[1]], axis=1, inplace=True)
shuffledFrame = shuffle(featureDataSet)
labels = shuffledFrame["label"].tolist()
values = shuffledFrame.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(values, labels, train_size=0.8, test_size=0.2)

classifier = SVC(kernel='linear', C=0.1)
#classifier = RandomForestClassifier(n_jobs=4)
#classifier = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=150)
#classifier = GradientBoostingClassifier(n_estimators=100, max_depth=5)
#classifier = DecisionTreeClassifier(max_depth=2, max_features=3)
#fastest algorithm
#classifier =  GaussianNB()
#classifier = KNeighborsClassifier(n_jobs=4, n_neighbors=5, algorithm='ball_tree')
#classifier = LogisticRegression()

#classifier = KMeans(n_clusters=2)
print("Classification start time: " + dt_string)
#classifier.fit(x_train, y_train)
validationUtils = CrossValidationUtils()
#paramObj = ParamSets()
hyperParamObj = HyperParameterSelectionUtils()
#hyperParamObj.perform_grid_search_param_selection(classifier, paramObj.ada_boost_params, values, labels)
#validationUtils.perform_kfold_cross_validation(10, classifier, values, labels)

classifier.fit(x_train, y_train)
classnames = ["fake news", "real news"]
#export_graphviz(classifier, filled=True, out_file="C:\Project\MIT_2020_FND\data\deTree.dot", class_names=classnames)

pred_time = d.datetime.now()
pred_string = pred_time.strftime("%d/%m/%Y %H:%M:%S")
print("Model Training and fitting done. Timestamp: " + str(pred_string))


classifierName = classifier.__class__.__name__
predicted_labels_x_test = classifier.predict(x_test)
y_predict = classifier.predict(x_test)
endTime = d.datetime.now()
end_string = timeStart.strftime("%d/%m/%Y %H:%M:%S")
print("Prediction end time: " + str(end_string))

print("Performance metrics for " + classifierName)
tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels_x_test).ravel()


print("")
print("Confusion Matrix Results: ")
print("True Negative: " + str(tn))
print("False Positive: " +str(fp))
print("False Negative: " +str(fn))
print("True Positive: " + str(tp))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn)
f_measure = (2 * (precision * recall)) / (precision + recall)
print("")

print("Precision: " + str(precision))
print("Accuracy: " + str(accuracy))
print("Recall: " + str(recall))
print("F-Measure: " + str(f_measure))

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
gp = GraphHelper()
#plotGraph = gp.create_learning_graph(graphTitle="Random Forest Classifier", estimator=classifier, X_values=x_train,
                                     #Y_Values=y_train, n_jobs=4, cv=None, axes=axes[:, 1])

gp.calculate_roc_graph(y_test, y_predict, ["fake", "real"], classifierName)
