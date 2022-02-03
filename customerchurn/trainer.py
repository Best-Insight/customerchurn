from customerchurn.data import get_data, clean_df, holdout
from customerchurn.model import get_model
from customerchurn.pipeline import get_pipeline
from customerchurn.mlflow import MLFlowBase
from customerchurn.params import MLFLOW_URI, EXPERIMENT_NAME

import joblib


class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)

    def train(self):

        model_name = "XXXX"

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", model_name)

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

        # train
        pipeline.fit(X_train, y_train)

        # make prediction for metrics
        y_pred = pipeline.predict(X_test)

        # evaluate metrics
        #SCORING FUNCTION
        #score = compute_rmse(y_pred, y_test)

        # save the trained model
        joblib.dump(pipeline, "model.joblib")

        # push metrics to mlflow
        self.mlflow_log_metric("score", score)

        # return the gridsearch in order to identify the best estimators and params
        return pipeline
