from customerchurn.data import get_data
from customerchurn.data import get_data_from_gcp, get_dataset_from_gcp, split_dataset, get_data_from_gcp_folder, star2recommend
from customerchurn.model import build_classifier_model, build_combined_class_model
# from customerchurn.pipeline import get_pipeline
from customerchurn.mlflowlog import MLFlowBase
from customerchurn.params import MLFLOW_URI, EXPERIMENT_NAME
from customerchurn.gcp import storage_upload_folder, storage_upload_file

import json

import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
# from tensorflow.keras.callbacks import EarlyStopping


class Trainer(MLFlowBase):

    def __init__(self, lr=0.001, batch_sizes=128):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)
        self.learning_rate = lr
        self.batch_sizes = batch_sizes

    def prediction_bias(self, y_pred, bias=0.5):
        y_pred[y_pred > bias] = 1
        y_pred[y_pred <= bias] = 0
        return y_pred

    def train(self):
        model_name = "yelp_score_10_midout"
        # create a mlflow training
        self.mlflow_create_run()
        # log params
        self.mlflow_log_param("customerchurn_gcp", model_name)
        # get data
        df = get_data_from_gcp(n_rows=None)
        df = star2recommend(df)
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
            monitor='val_loss', restore_best_weights=True, patience=3)

        history = model.fit(X_train,
                            y_train,
                            batch_size=128,
                            epochs=100,
                            validation_split=0.2,
                            callbacks=[es],
                            verbose=1)

        with open('yelp_score_10_midout_history.json', 'w') as fp:
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
        # y_pred = model.predict(X_test)

        # y_pred = self.prediction_bias(y_pred, 0.5)

        # matrix = confusion_matrix(y_test, y_pred.astype(int))
        # self.mlflow_log_metric('confusion_matrix', matrix)
        # evaluate metrics
        #SCORING FUNCTION
        # score = compute_rmse(y_pred, y_test)

        # save the trained model
        model.save('yelp_score_10_midout_model')

        # push metrics to mlflow
        # self.mlflow_log_metric("score", score)

        # return the gridsearch in order to identify the best estimators and params
        return model

    # def train_dataset(self):
    #     model_name = "customerchurn_gcp"
    #     # create a mlflow training
    #     self.mlflow_create_run()
    #     # log params
    #     self.mlflow_log_param("customerchurn_gcp", model_name)
    #     # get data
    #     ds = get_dataset_from_gcp()
    #     # train test split
    #     train_dataset, val_dataset, test_dataset = split_dataset(ds)
    #     # create model
    #     model = build_classifier_model(self.learning_rate)

    #     es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                         restore_best_weights=True,
    #                                         patience=2)

    #     history = model.fit(train_dataset,
    #                         epochs=100,
    #                         validation_data=val_dataset,
    #                         callbacks=[es],
    #                         verbose=1)

    #     with open('my_model_history.json', 'w') as fp:
    #         json.dump(history.history, fp)

    #     # evaluation = model.evaluate(test_dataset)
    #     # self.mlflow_log_param("evaluation", evaluation)

    #     # y_pred = model.predict(X_test)

    #     # y_pred = self.prediction_bias(y_pred, 0.5)

    #     # matrix = confusion_matrix(y_test, y_pred.astype(int))
    #     # self.mlflow_log_metric('confusion_matrix', matrix)
    #     # evaluate metrics
    #     #SCORING FUNCTION
    #     # score = compute_rmse(y_pred, y_test)

    #     # save the trained model
    #     model.save('real_estate_model')

    #     # push metrics to mlflow
    #     # self.mlflow_log_metric("score", score)

    #     # return the gridsearch in order to identify the best estimators and params
    #     return model

    def combined_train(self):
        model_name = "yelp_score_5"
        # create a mlflow training
        self.mlflow_create_run()
        # log params
        self.mlflow_log_param("customerchurn_gcp", model_name)
        # get data
        df = get_data_from_gcp(n_rows=None)
        df = star2recommend(df)
        # get x, y
        X = df[['review', 'score']]
        y = df['recommendation'].apply(lambda x: 0
                                       if x == 'Not Recommended' else 1)
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.33,
                                                            random_state=42)
        X_train_text = X_train['review']
        X_train_num = X_train['score']
        X_test_text = X_test['review']
        X_test_num = X_test['score']
        # create model
        model = build_combined_class_model(self.learning_rate)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              restore_best_weights=True,
                                              patience=3)

        history = model.fit([X_train_text, X_train_num],
                            y_train,
                            batch_size=128,
                            epochs=100,
                            validation_split=0.2,
                            callbacks=[es],
                            verbose=1)

        with open('yelp_score_5_history.json', 'w') as fp:
            json.dump(history.history, fp)

        evaluation = model.evaluate([X_test_text, X_test_num], y_test)
        self.mlflow_log_param("evaluation", evaluation)

        # save the trained model
        model.save('yelp_score_5_model')

        # push metrics to mlflow
        # self.mlflow_log_metric("score", score)

        # return the gridsearch in order to identify the best estimators and params
        return model




if __name__ == '__main__':

    model = Trainer()
    model = model.train()
    # model = model.combined_train()
    storage_upload_folder(rm=False,
                          folder_name='yelp_score_10_midout_model',
                          gcp_folder='models')
    storage_upload_file(rm=False,
                        file_name='yelp_score_10_midout_history.json',
                        gcp_folder='history')
