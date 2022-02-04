from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from customerchurn.preprocessing import series_prepro

def create_pre_pipe():
    prep = FunctionTransformer(series_prepro)

    preprocessor = ColumnTransformer([('text_nltk', prep, 'review')],
                                    remainder='passthrough')
    pipe = make_pipeline(preprocessor)

    return pipe
