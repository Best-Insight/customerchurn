import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
# from tensorflow.keras.optimizers import Adam


tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

def build_classifier_model(lr):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')

    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']

    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)

    model = tf.keras.Model(text_input, net)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])

    return model




# def get_model(model_name):

#     if model_name == "RNN-V1":

#         # model_params = dict(
#         #   )

#         # model = RandomForestRegressor()
#         # model.set_params(**model_params)

#         # return model
#         pass

#     elif model_name == "RNN-V2":

#         # model_params = dict(
#         #   )

#         # model = RandomForestRegressor()
#         # model.set_params(**model_params)

#         # return model
#         pass
