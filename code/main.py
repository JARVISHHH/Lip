import os
import sys
import argparse
import re
import tensorflow as tf
from datetime import datetime

import hyperparameters as hp 
from preprocess.dataset import GRIDDataset
from tensorboard_utils import \
        CustomModelSaver
from models import build_model

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

def train(model, train_datasets, val_datasets,checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=logs_path,
        #     update_freq='batch',
        #     profile_batch=0),
        # # ImageLabelingLogger(logs_path, datasets),
        # CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights),
    ]

    # Begin training
    model.fit(
        train_datasets,
        epochs=hp.num_epochs,
        validation_data=val_datasets,
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
        train_datasets = GRIDDataset(hp.data_path, 'train').build()
        val_datasets = GRIDDataset(hp.data_path, 'val').build()
        model = build_model()
        checkpoint_path = "checkpoints" + os.sep + \
            "LipNet" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "LipNet" + \
            os.sep + timestamp + os.sep

        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    train(model, train_datasets, val_datasets, checkpoint_path, logs_path, init_epoch)

ARGS = parse_args()

main()