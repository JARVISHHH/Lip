import tensorflow as tf
import keras
from keras.layers import \
       Conv3D, MaxPool3D, TimeDistributed, Dropout, Flatten, Dense, LSTM, Bidirectional, Activation, ZeroPadding3D, MaxPooling3D, GRU

import hyperparameters as hp

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model():
    input = keras.layers.Input(shape=hp.input_shape, name='input')

    architecture = [
            ZeroPadding3D(padding=(1, 2, 2), name='zero1'),
            Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', kernel_initializer='he_normal', name='conv1'),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1'),
            Dropout(0.5),

            ZeroPadding3D(padding=(1, 2, 2), name='zero2'),
            Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2'),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'),
            Dropout(0.5),

            ZeroPadding3D(padding=(1, 1, 1), name='zero3'),
            Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv3'),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'),
            Dropout(0.5),

            TimeDistributed(Flatten()),

            Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat'),
            Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat'),

            Dense(hp.output_size, kernel_initializer='he_normal', name='dense1'),
            Activation('softmax', name='softmax')
        ]

    output = input
    for layer in architecture:
        output = layer(output)
    
    model = keras.Model(input, output, name='LipNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    model.compile(optimizer=optimizer, 
                  loss=CTCLoss,
                #   metrics=["accuracy"]
                  )

    return model