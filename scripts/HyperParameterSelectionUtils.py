import sklearn.model_selection as ms


def perform_grid_search_param_selection(estimator, param_dictionary, x_dataset, y_labels):
    """This function performs the Grid Search technique, given a set of parameters
    Parameters
    --------------
    estimator: A pre-built machine learning SciKit-Learn model
    param_dictionary: A JSON style object containing hyper-parameter and values to test
    x_dataset: The dataset to fit model
    y_labels: A collection of labels for each sample in the x_dataset
    """
    "define the Grid Search Model"
    grid_searcher = ms.GridSearchCV(estimator=estimator, param_grid=param_dictionary, scoring="accuracy",
                                    n_jobs=4, cv=3, return_train_score=True)
    "fit the model with features and labels data - may take a while depending on model type and data sizes"
    grid_searcher.fit(x_dataset, y_labels)

    "get the best parameter configurations"
    print(grid_searcher.best_params_)
    print("Best parameter set score: " + str(grid_searcher.best_score_))


class HyperParameterSelectionUtils:
    pass
