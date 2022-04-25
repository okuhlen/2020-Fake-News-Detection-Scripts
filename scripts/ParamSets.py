class ParamSets:
    random_forest_params = [{
        'n_estimators': [2, 5, 7, 10],
        'max_depth': [0, 1, 3, 5, 7]
    }]

    xgboost_params = [
        {'n_estimators': [7, 10, 50, 100],
         'max_depth': [1, 3, 5, 7]
         }]

    svm_params = [{
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
    }]

    ada_boost_params = [
        {
            'n_estimators': [20, 50, 100, 150],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    ]

    knn_model_params = [{
        'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    }]

    log_regression_params = [{
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'saga']
    }]

    decision_tree_set_params = [{
        ''
    }]

    def get_random_forest_param_set(self):
        return self.random_forest_params

    def get_knn_param_set(self):
        return self.knn_model_params

    def get_logistic_regresssion_set(self):
        return self.log_regression_params

    def get_decision_tree_param_set(self):
        return self.decision_tree_set_params