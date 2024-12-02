import tensorflow as tf
from keras.layers import \
       Conv3D, MaxPool3D, TimeDistributed, Dropout, Flatten, Dense, LSTM, Bidirectional, Activation

import hyperparameters as hp
import preprocess as pp

class LipNet(tf.keras.Model):
    def __init__(self):
        super(LipNet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.architecture = [
            Conv3D(128, 3, padding='same'),
            Activation('relu'),
            MaxPool3D((1, 2, 2)),

            Conv3D(256, 3, padding='same'),
            Activation('relu'),
            MaxPool3D((1, 2, 2)),

            Conv3D(75, 3, padding='same'),
            Activation('relu'),
            MaxPool3D((1, 2, 2)),
            
            TimeDistributed(Flatten()),

            Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            Dropout(.5),

            Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            Dropout(.5),
			
			# Dense layer with the number of classes as the number of neurons
			Dense(pp.char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(y_true, y_pred):
        """ Loss function for the model. """

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss