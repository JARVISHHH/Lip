import os
import tensorflow as tf
import pathlib

num_epochs = 10

learning_rate = 0.001

max_num_weights = 5

minibatch_size = 5

data_path = pathlib.Path('../datasets')

face_predictor_path = pathlib.Path('../common/predictors/shape_predictor_68_face_landmarks.dat')

absolute_max_string_len = 32

data_path = os.path.normpath('../datasets')

face_predictor_path = os.path.normpath('../common/predictors/shape_predictor_68_face_landmarks.dat')

absolute_max_string_len = 32

decay_rate = 0.925
base_flip_prob = 0.2
upperbond_flip_prob = 0.5
base_jitter_prob = 0.01
upperbond_jitter_prob = 0.05
base_gaussian_prob = 0.05
upperbond_gaussian_prob = 0.1


##### DO NOT CHNGE #####

frames_number = 75
image_width = 100
image_height = 50
image_channel = 3
input_shape = (frames_number, image_width, image_height, image_channel)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="", num_oov_indices=0)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True, num_oov_indices=0)

output_size = char_to_num.vocabulary_size() + 1