import os
import sys
import argparse
import re
import tensorflow as tf
from datetime import datetime
import numpy as np
import glob

import hyperparameters as hp 
from preprocess.curriculum import Curriculum, CurriculumUpdateCallback
from preprocess.dataset import GRIDDataset
from preprocess.video import Video
from preprocess.visualization import show_video_subtitle
from tensorboard_utils import \
        CustomModelSaver
from models import build_model

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--load-cache',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='''Load the dataset cache.''')
    parser.add_argument(
        '--load-testcache',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='''Load the test dataset cache.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--load-pretrain',
        default=None,
        help='''Path to model pretrain weights file (should end with the
        extension .h5).''')
    parser.add_argument(
        '--train',
        action='store_true',
        help='''Train.''')
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
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',  # Save histograms of weights for each epoch
            write_graph=True,  # Save the computation graph
            write_images=True  # Save images of weights
        ),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]
    
    if train_datasets.curriculum :
        curriculum_callback = CurriculumUpdateCallback(train_datasets.curriculum)
        callback_list.append(curriculum_callback)

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
    
    #Defined dynamically curriculum rules for data augmention
    def curriculum_rules(epoch):
        flip_prob = min(hp.base_flip_prob + epoch * 0.05, hp.upperbond_flip_prob) 
        jitter_prob = min(hp.base_jitter_prob + epoch * 0.01, hp.upperbond_jitter_prob)
        gaussian_prob = min(hp.base_gaussian_prob + epoch * 0.01, hp.upperbond_gaussian_prob)
        decay_rate = hp.decay_rate ** epoch
        
        if epoch < 1:
            return { 'sentence_length': 1, 'decay_rate': decay_rate}
        elif 1 <= epoch < 2:
            return { 'sentence_length': 2, 'decay_rate': decay_rate}
        elif 2 <= epoch < 3:
            return { 'sentence_length': 2, 'flip_probability': flip_prob, 'gaussian_probability': gaussian_prob, 'decay_rate': decay_rate }
        elif 3 <= epoch < 4:
            return { 'sentence_length': 3, 'flip_probability': flip_prob, 'jitter_probability': jitter_prob, 'gaussian_probability': gaussian_prob, 'decay_rate': decay_rate}
        elif 4 <= epoch < 5:
            return { 'sentence_length': -1, 'gaussian_probability': gaussian_prob, 'decay_rate': decay_rate }
        elif 5 <= epoch < 6:
            return { 'sentence_length': -1, 'flip_probability': flip_prob, 'gaussian_probability': gaussian_prob, 'decay_rate': decay_rate }
        return { 'sentence_length': -1, 'flip_probability': flip_prob, 'jitter_probability': jitter_prob, 'gaussian_probability': gaussian_prob, 'decay_rate': decay_rate }

    """ Main function. """
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0
    #Initialize the curriculum for augmenation
    curriculum = Curriculum(curriculum_rules)

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    
    checkpoint_path = "checkpoints" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + timestamp + os.sep

    model = build_model()
    model.summary()

    if ARGS.load_pretrain is not None:
        model.load_weights(ARGS.load_pretrain, by_name=False)
        print("Loaded weights " + ARGS.load_pretrain)
    # Load checkpoints
    elif ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)
        print("Loaded weights " + ARGS.load_checkpoint)

    # Make checkpoint directory if needed
    if ARGS.train and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Train
    if ARGS.train:
        train_datasets = GRIDDataset(hp.data_path, 'train', ARGS.load_cache, ARGS.load_testcache, curriculum=curriculum).build()
        val_datasets = GRIDDataset(hp.data_path, 'val', ARGS.load_cache, ARGS.load_testcache).build()
        
        train(model, train_datasets, val_datasets, checkpoint_path, logs_path, init_epoch)
    # Test
    if ARGS.evaluate:
        test_path = os.path.join(hp.data_path, "tests", "*")
        for video_path in glob.glob(test_path):
            print("Predicting test data " + video_path + "...")
            try:
                if os.path.isfile(video_path):
                    video = Video('face').from_video(video_path)
                else:
                    video = Video('mouth').from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print("Error loading video: " + video_path)
                continue
            if (video.data.shape != hp.input_shape):
                print("Video " + video_path + " has incorrect shape " + str(video.data.shape) + ", must be " + str(hp.input_shape) + "")
                continue
            X_data       = np.array([video.data]).astype(np.float32) / 255
            input_length = np.array([len(video.data)])
            y_pred       = model.predict(X_data)
            decoded      = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=False)[0][0].numpy()

            text = "".join([bytes.decode(x) for x in hp.num_to_char(decoded[0]).numpy()])
            # text = tf.strings.reduce_join([bytes.decode(x) for x in hp.num_to_char(decoded[0]).numpy()])
            print(text)

            # show_video_subtitle(video.face, decoded)

ARGS = parse_args()

main()