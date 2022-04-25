from datetime import datetime

from sklearn.model_selection import cross_val_score


class CrossValidationUtils:

    def perform_kfold_cross_validation(self, numFolds, classifier, x_dataset, y_dataset):
        """This function performs cross validation for a given classification model.
        Parameters
        -------------

        numFolds: The number of folds for the cross validation
        classifier: The SciKit-Learn machine learning model
        x_dataset: The feature set be used for cross validation (features).
        y_dataset: labels for each sample in the provided feature set"""
        crossValScore = cross_val_score(classifier, x_dataset, y_dataset, cv=numFolds, n_jobs=4)
        print("K-Fold Cross Validation Complete for "+classifier.__class__.__name__)

        counter = 1
        for i in crossValScore:
            """Output score for each fold"""
            print("Fold " + str(counter) + " Cross Validation Score: " + str(i))
            counter = counter + 1
