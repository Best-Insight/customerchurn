import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# from tensorflow.keras.optimizers import Adam

### ----------------------------------------------------------------------------
### bert model part
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
    ## choose output
    # net = outputs['pooled_output']
    net = outputs['sequence_output']

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
    ## choose output
    # net = outputs['pooled_output']
    net = outputs['sequence_output']

    net = tf.keras.layers.LSTM(16, activation="tanh")(net)
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
    output = tf.keras.layers.Dense(8, activation='relu')(combined)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

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

### ----------------------------------------------------------------------------
### auto encoder model part
def build_encoder(latent_dimension):
    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.Masking(mask_value=0))
    encoder.add(
        tf.keras.layers.LSTM(16, activation="tanh", return_sequences=True))
    encoder.add(tf.keras.layers.LSTM(latent_dimension, activation="tanh"))
    return encoder


def build_decoder(n_words):
    # $CHALLENGIFY_BEGIN
    decoder = tf.keras.Sequential()
    decoder.add(RepeatVector(n_words))  #RepeatVector add comments
    decoder.add(
        tf.keras.layers.LSTM(50,
                             activation='tanh',
                             return_sequences=True))
    decoder.add(TimeDistributed(
        tf.keras.layers.Dense(50)))  #TimeDistrivuted add comments
    return decoder


def build_autoencoder(latent_dimension, n_words):
    inp = tf.keras.layers.Input(n_words)
    encoder = build_encoder(latent_dimension)
    encoded = encoder(inp)
    decoder = build_decoder(n_words)
    decoded = decoder(encoded)
    autoencoder = tf.keras.models.Model(inp, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,
                                                           beta_1=0.9,
                                                           beta_2=0.999,
                                                           epsilon=1e-08,
                                                           decay=0.0),
                        loss='mse',
                        metrics=['mse'])
    return autoencoder
