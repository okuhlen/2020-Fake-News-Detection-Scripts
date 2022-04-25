import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
from sklearn.preprocessing import label_binarize


class GraphHelper:

    def create_learning_graph(self, estimator, graphTitle, X_values, Y_Values, y_limits=None, cv = None, axes=None,
                              n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

        if axes is None:
            axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title=graphTitle
        if y_limits is not None:
            axes[0].set_ylim(*y_limits)
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X_values, Y_Values, cv,
                                                                            n_jobs=n_jobs,
                                                                train_sizes=train_sizes, return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

    def calculate_roc_graph(self, y_test, y_predict, labels, classifierName):
        print("Now generating graph for " + classifierName)
        y_pred = np.array(y_predict)

        binLabels = label_binarize(y_test, classes=[0,1])
        numClasses = binLabels.shape[1]

        tpr = dict()
        fpr = dict()
        thresholds = None
        roc_auc = dict()
        print("")
        print(y_pred.ravel())
        print("ROC AUC Score: " + str(roc_auc_score(y_test, y_pred.ravel())))
        print(numClasses)
        for i in range(numClasses):

            fpr[i], tpr[i], thresholds = roc_curve(y_test, y_pred.ravel())
            roc_auc[i] = auc(fpr[i], tpr[i])

        lw=3
        plt.figure()
        plt.plot(fpr[0], tpr[0], color='orange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Graph for ' + classifierName)
        plt.legend(loc="lower right")
        plt.show()