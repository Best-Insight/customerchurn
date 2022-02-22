from customerchurn.data import get_data
from customerchurn.data import get_data_from_gcp
from customerchurn.model import build_classifier_model
# from customerchurn.pipeline import get_pipeline
from customerchurn.mlflowlog import MLFlowBase
from customerchurn.params import MLFLOW_URI, EXPERIMENT_NAME
from customerchurn.gcp import history_upload, storage_upload

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from tensorflow.keras.callbacks import EarlyStopping


class Trainer(MLFlowBase):

    def __init__(self, lr=0.001, batch_sizes=128):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)
        self.learning_rate = lr
        self.batch_sizes = batch_sizes

    def train(self):

        model_name = "customerchurn_gcp"

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("customerchurn_gcp", model_name)

        # get data
        df = get_data_from_gcp(n_rows= None)

        # get x, y
        X = df['review']
        y = df['recommendation'].apply(lambda x: 0
                                            if x == 'Not Recommended' else 1)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.33,
                                                            random_state=42)

        # create model


        model = build_classifier_model(self.learning_rate)

        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', restore_best_weights=True, patience=2)

        history = model.fit(X_train,y_train, batch_size =128 , epochs =100,
                            validation_split=0.3, callbacks=[es], verbose = 1)

        import json
        with open ('my_model_history.json', 'w') as fp:
            json.dump(history.history, fp)

        evaluation = model.evaluate(X_test, y_test)
        self.mlflow_log_param("evaluation", evaluation)


        # model = KerasClassifier(build_fn=build_classifier_model,
        #                         validation_split=0.2,
        #                         callbacks=[es])
        # KerasRegressor(build_fn=baseline_model_v2,
        #                verbose=1,
        #                epochs=1000,
        #                batch_size=128,
        #                shuffle=True,
        #                validation_split=0.2,
        #                callbacks=[es])

        # # log params
        # self.mlflow_log_param("learning_rate", self.learning_rate)

        # # create pipeline
        # pipeline = get_pipeline(model)
        # learning_rate = self.learning_rate
        # batch_size = self.batch_sizes
        # param_grid = dict(batch_size=batch_size, lr=learning_rate)
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

        # # train
        # grid.fit(X_train, y_train)


        # best_model = grid.best_estimator_



        # make prediction for metrics
        y_pred = model.predict(X_test)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0


        matrix =confusion_matrix(y_test, y_pred.astype(int))
        self.mlflow_log_metric('confusion_matrix', matrix)
        # evaluate metrics
        #SCORING FUNCTION
        # score = compute_rmse(y_pred, y_test)

        # save the trained model
        model.save('my_model')

        # push metrics to mlflow
        # self.mlflow_log_metric("score", score)

        # return the gridsearch in order to identify the best estimators and params
        return model







if __name__ == '__main__':

    model = Trainer()
    model = model.train()
    storage_upload()
    history_upload()
