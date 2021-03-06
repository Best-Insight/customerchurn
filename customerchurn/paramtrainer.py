from customerchurn.data import get_data, clean_df, holdout
from customerchurn.model import get_model
from customerchurn.pipeline import get_pipeline
from customerchurn.mlflowlog import MLFlowBase
from customerchurn.params import MLFLOW_URI, EXPERIMENT_NAME
from sklearn.model_selection import GridSearchCV

import joblib


class ParamTrainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)

    def train(self, params):

        # results
        models = {}

        # iterate on models
        for model_name, model_params in params.items():

            # create a mlflow training
            self.mlflow_create_run()

            # log params
            self.mlflow_log_param("model_name", model_name)
            for key, value in model_params.items():
                self.mlflow_log_param(key, value)

            # get data
            df = get_data(path = './customerchurn/data/concat.csv')
            df = clean_df(df)

            # holdout
            X_train, X_test, y_train, y_test = holdout(df)

            # log params
            self.mlflow_log_param("model", model_name)

            # create model
            model = get_model(model_name)

            # create pipeline
            pipeline = get_pipeline(model)

            # create gridsearch object
            grid_search = GridSearchCV(
                pipeline,
                param_grid=model_params,
                cv=5)

            # train with gridsearch
            grid_search.fit(X_train, y_train)

            # score gridsearch
            score = grid_search.score(X_test, y_test)

            # save the trained model
            #joblib.dump(pipeline, f"{model_name}.joblib")

            # push metrics to mlflow
            self.mlflow_log_metric("score", score)

            # return the gridsearch in order to identify the best estimators and params
            models[model_name] = grid_search

        return models
