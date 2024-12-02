import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
import hyperparameters as hp

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# TODO: double check the magic numbers
class GRIDDataset():

    def __init__(self, data_path):
        self.data_path = data_path

        self.task = 'LipNet'

        self.data = tf.data.Dataset.list_files(self.data_path)
        self.data = self.data.shuffle(500, reshuffle_each_iteration=False)  # Shuffle the data 500 by 500
        self.data = self.data.map(self.mappable_function)
        self.data = self.data.padded_batch(hp.batch_size, padded_shapes=([75, None, None, None], [75]))  # make sure each video has 75 frames and 40 aligns
        self.data = self.data.prefetch(tf.data.AUTOTUNE)
        # Added for split 
        self.test_data = self.data.take(hp.test_size)
        self.train_data = self.data.skip(hp.test_size)

    def load_video(self, path: str) -> List[float]: 

        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
            ret, frame = cap.read()
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236, 80:220, :])  # TODO: hardcoded, change it to dlib detector
        cap.release()
        
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        return tf.cast((frames - mean), tf.float32) / std

    def load_alignments(self, path: str) -> List[str]: 
        with open(path, 'r') as f: 
            lines = f.readlines() 
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil': 
                tokens = [*tokens, ' ', line[2]]
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

    def load_data(self, path: str): 
        path = bytes.decode(path.numpy())
        # File name splitting for Mac
        # speaker_name = path.split('/')[-2]
        # file_name = path.split('/')[-1].split('.')[0]
        # File name splitting for Windows
        speaker_name = path.split('\\')[-2]
        file_name = path.split('\\')[-1].split('.')[0]
        video_path = os.path.join('datasets', 'GRID', 'videos', speaker_name, f'{file_name}.mpg')
        alignment_path = os.path.join('datasets', 'GRID', 'alignments', speaker_name, f'{file_name}.align')
        frames = self.load_video(video_path)
        alignments = self.load_alignments(alignment_path)
        
        return frames, alignments

    def mappable_function(self, path:str) -> List[str]:
        result = tf.py_function(self.load_data, [path], (tf.float32, tf.int64))
        return result