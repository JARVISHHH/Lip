import keras.backend
import tensorflow as tf
import keras
from keras.layers import \
       Conv3D, TimeDistributed, Dropout, Flatten, Dense, Bidirectional, Activation, ZeroPadding3D, MaxPooling3D, GRU, BatchNormalization, SpatialDropout3D, MaxPool3D, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.regularizers import l2
import hyperparameters as hp

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * keras.backend.cast(input_shape[1], 'float32')

        decode, log = keras.backend.ctc_decode(y_pred, input_length, greedy=True)
        decode = keras.backend.ctc_label_dense_to_sparse(decode[0], keras.backend.cast(input_length, 'int32'))
        y_true_sparse = keras.backend.ctc_label_dense_to_sparse(y_true, keras.backend.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        # Convert len(y_true) to float32
        self.counter.assign_add(tf.cast(len(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Error Rate
    """
    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * keras.backend.cast(input_shape[1], 'float32')

        decode, log = keras.backend.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)

        decode = keras.backend.ctc_label_dense_to_sparse(decode[0], keras.backend.cast(input_length, 'int32'))
        y_true_sparse = keras.backend.ctc_label_dense_to_sparse(y_true, keras.backend.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(tf.cast(len(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    try:
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    except:
        print(y_pred)
        print(y_true)
        raise ValueError()

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model():
    input = keras.layers.Input(shape=hp.input_shape, name='input')

    architecture = [
            ##### GitHub model 1 #####
            ## aorund 4 million parameters
            # ZeroPadding3D(padding=(1, 2, 2), name='zero1'),
            # Conv3D(32, (3, 5, 5), strides=(1, 2, 2), activation='relu', kernel_initializer='he_normal', name='conv1'),
            # MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1'),
            # Dropout(0.5),

            # ZeroPadding3D(padding=(1, 2, 2), name='zero2'),
            # Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv2'),
            # MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'),
            # Dropout(0.5),

            # ZeroPadding3D(padding=(1, 1, 1), name='zero3'),
            # Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal', name='conv3'),
            # MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'),
            # Dropout(0.5),

            # TimeDistributed(Flatten()),

            # Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat'),
            # Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat'),

            # Dense(hp.output_size, kernel_initializer='he_normal', name='dense1'),
            # Activation('softmax', name='softmax')

            ##### GitHub model 2 #####
            ## aorund 4 million parameters
            ZeroPadding3D(padding=(1, 2, 2), name='zero1'),
            Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1'),
            BatchNormalization(name='batc1'),
            Activation('relu', name='actv1'),
            SpatialDropout3D(0.5),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1'),

            ZeroPadding3D(padding=(1, 2, 2), name='zero2'),
            Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2'),
            BatchNormalization(name='batc2'),
            Activation('relu', name='actv2'),
            SpatialDropout3D(0.5),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'),

            ZeroPadding3D(padding=(1, 1, 1), name='zero3'),
            Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3'),
            BatchNormalization(name='batc3'),
            Activation('relu', name='actv3'),
            SpatialDropout3D(0.5),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'),

            TimeDistributed(Flatten()),

            Bidirectional(GRU(256, return_sequences=True, reset_after=False, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat'),
            Bidirectional(GRU(256, return_sequences=True, reset_after=False, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat'),

            Dense(hp.output_size, kernel_initializer='he_normal', name='dense1'),
            Activation('softmax', name='softmax')

            ##### Youtube model #####
            ## aorund 7 million parameters
            # Conv3D(128, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # Conv3D(256, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # Conv3D(75, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # TimeDistributed(Flatten()),

            # Bidirectional(GRU(256, return_sequences=True, reset_after=False, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat'),
            # Bidirectional(GRU(256, return_sequences=True, reset_after=False, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat'),

            # Dense(hp.output_size, kernel_initializer='he_normal', name='dense1')

            ##### Youtube model #####
            ## aorund 4 million parameters
            # Conv3D(128, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # Conv3D(256, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # Conv3D(75, 3, padding='same'),
            # Activation('relu'),
            # MaxPool3D((1,2,2)),

            # TimeDistributed(Flatten()),

            # Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            # Dropout(.5),

            # Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            # Dropout(.5),

            # Dense(hp.output_size, kernel_initializer='he_normal', activation='softmax')


            
            # Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding='same', 
            # kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
            # BatchNormalization(),
            # Activation('relu'),
            # SpatialDropout3D(0.5),
            # MaxPooling3D(pool_size=(2, 2, 2)),
            
            # Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal'),
            # BatchNormalization(),
            # Activation('relu'),
            # SpatialDropout3D(0.5),
            # MaxPooling3D(pool_size=(2, 2, 2)),

            # Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal'),
            # BatchNormalization(),
            # Activation('relu'),
            # SpatialDropout3D(0.5),
            # MaxPooling3D(pool_size=(2, 2, 2)),

            # # Flatten
            # TimeDistributed(Flatten()),

            # Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal')),
            # Dropout(0.5),
            # Bidirectional(GRU(64, return_sequences=False, kernel_initializer='Orthogonal')),
            # Dropout(0.5),

            # # Fully connected layers

            # Dense(hp.output_size, kernel_initializer='he_normal', activation='softmax')

    ]

    output = input
    for layer in architecture:
        output = layer(output)
    
    


    model = keras.Model(input, output, name='LipNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    model.compile(optimizer=optimizer, 
                  loss=CTCLoss,
                   metrics= [CERMetric(name="CER"), WERMetric(name="WER")]
                  )

    return model



# def sparse_categorical_accuracy_fn(y_true, y_pred):
#     y_pred = tf.argmax(y_pred, axis=-1)  # Reduce the last dimension
#     return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def get_callbacks():
    checkpoint_path = "checkpoints/lipnet_model_{epoch:02d}_{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                 monitor='val_loss', 
                                 save_best_only=True, 
                                 save_weights_only=False, 
                                 verbose=1)

    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, update_freq='epoch')

    return [checkpoint, tensorboard]