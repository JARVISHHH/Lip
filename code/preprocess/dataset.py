import os
import tensorflow as tf
import numpy as np
import hyperparameters as hp
import pickle
import glob
import multiprocessing

from preprocess.video import Video
from preprocess.align import Align
from preprocess.helpers import text_to_labels, get_list_safe

class GRIDDataset(tf.keras.utils.Sequence):

    def __init__(self, data_path, folder_name):
        self.video_input_shape = hp.input_shape
        
        self.data_path = data_path
        self.video_path = os.path.join(self.data_path, "mouth", folder_name)
        self.align_path = os.path.join(self.data_path, "alignments")
        self.cache_path = os.path.join(self.data_path, folder_name + ".cache")

        self.task = 'LipNet'

        self.cur_train_index = multiprocessing.Value('i', 0)
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = None  # TODO: change this after adding augmentation
        self.random_seed     = 13
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch   = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self):
        self.build_dataset()
        return self

    @property
    def video_size(self):
        return len(self.video_list)

    @property
    def default_steps(self):
        return self.video_size / hp.minibatch_size
    
    def get_align(self, _id):
        return self.align_hash[_id]

    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):
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
            if (video.data.shape != self.video_input_shape):
                print("Video " + video_path + " has incorrect shape " + str(video.data.shape) + ", must be " + str(self.video_input_shape) + "")
                continue
            video_list.append(video_path)
        return video_list
    
    def enumerate_aligns(self, video_list):
        align_hash = {}
        for video_path in video_list:
            speaker = video_path.split(os.sep)[-2]
            video_id = video_path.split(os.sep)[-1]
            align_path = os.path.join(self.align_path, speaker, video_id) + ".align"
            align_hash[os.path.join(speaker, video_id)] = Align(text_to_labels).from_file(align_path)
        return align_hash
    
    def build_dataset(self):
        if (os.path.isfile(self.cache_path)):
            print("\nLoading dataset list from cache...")
            with open (self.cache_path, 'rb') as fp:
                self.video_list, self.align_hash = pickle.load(fp)
        else:
            print("\nEnumerating dataset list from disk...")
            # TODO: decide the data root
            self.video_list = self.enumerate_videos(os.path.join(self.video_path, '*', 'b*'))
            self.align_hash = self.enumerate_aligns(self.video_list)
            with open(self.cache_path, 'wb') as fp:
                pickle.dump((self.video_list, self.align_hash), fp)

        print("Found {} videos.\n".format(len(self.video_list)))

        np.random.shuffle(self.video_list)

    def get_batch(self, start_index, size):
        video_list = self.video_list

        X_data_path = video_list[start_index : start_index + size]
        X_data = []
        Y_data = []
        for path in X_data_path:
            video = Video('mouth').from_frames(path)
            align = self.get_align(os.path.join(path.split(os.sep)[-2], path.split(os.sep)[-1]))
            video_unpadded_length = video.length
            if self.curriculum is not None:
                video, align, video_unpadded_length = self.curriculum.apply(video, align)
            X_data.append(video.data)
            Y_data.append(align.padded_label)

        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        return X_data, Y_data

    def __getitem__(self, index):

        return self.get_batch(index * hp.minibatch_size, hp.minibatch_size)
    
    def __len__(self):
        return self.video_size // hp.minibatch_size