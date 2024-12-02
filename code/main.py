import os
import sys
import argparse
import re
import tensorflow as tf
from datetime import datetime

import hyperparameters as hp
from models import LipNet
from preprocess import GRIDDataset
from tensorboard_utils import \
        CustomModelSaver

from keras.models import Sequential 
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
import preprocess as pp

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        choices=['LipNet'],
        default='LipNet',
        help='''Which task of the assignment to run -
        LipNet.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        # ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if ARGS.task == 'LipNet':
        datasets = GRIDDataset(hp.data_path)
        model = LipNet()
        model(tf.keras.Input(shape=hp.input_shape))
        checkpoint_path = "checkpoints" + os.sep + \
            "LipNet" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "LipNet" + \
            os.sep + timestamp + os.sep
        
        # model = Sequential()
        # model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPool3D((1,2,2)))

        # model.add(Conv3D(256, 3, padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPool3D((1,2,2)))

        # model.add(Conv3D(75, 3, padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPool3D((1,2,2)))

        # model.add(TimeDistributed(Flatten()))

        # model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        # model.add(Dropout(.5))

        # model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        # model.add(Dropout(.5))

        # model.add(Dense(pp.char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

        # Print summary of model
        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"],
        run_eagerly=True)

    train(model, datasets, checkpoint_path, logs_path, init_epoch)

ARGS = parse_args()

main()