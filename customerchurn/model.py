import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
# from tensorflow.keras.optimizers import Adam

tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'


def build_classifier_model(lr):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess,
                                         name='preprocessing')

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder,
                             trainable=False,
                             name='BERT_encoder')

    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']

    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid',
                                name='classifier')(net)

    model = tf.keras.Model(text_input, net)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[
                      'accuracy',
                      tf.keras.metrics.Recall(),
                      tf.keras.metrics.Precision(),
                      tf.keras.metrics.AUC()
                  ])

    return model


def build_nlp_layer():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess,
                                         name='preprocessing')

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder,
                             trainable=False,
                             name='BERT_encoder')

    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']

    net = tf.keras.layers.Dropout(0.1)(net)

    nlp_layer = tf.keras.models.Model(text_input, net)

    return nlp_layer


def build_num_layer():
    num_input = tf.keras.layers.Input(shape=(1, ), name='number')
    x_num = tf.keras.layers.Dense(16, activation="relu")(num_input)
    x_num = tf.keras.layers.Dropout(0.2)(x_num)
    num_layer = tf.keras.models.Model(num_input, x_num)
    return num_layer


def build_combined_class_model(lr):
    nlp_layer = build_nlp_layer()
    num_layer = build_num_layer()

    combined = tf.keras.layers.concatenate(
        [nlp_layer.output, num_layer.output])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    model_combined = tf.keras.models.Model(
        inputs=[nlp_layer.input, num_layer.input], outputs=output)

    model_combined.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC()
        ])

    return model_combined

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
