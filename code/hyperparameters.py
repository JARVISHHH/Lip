import os

num_epochs = 10

learning_rate = 0.001

max_num_weights = 5

minibatch_size = 5

frames_number = 75
image_width = 100
image_height = 50
image_channel = 3
input_shape = (frames_number, image_width, image_height, image_channel)
output_size = 28  # 28 letters + space + CTC space

data_path = os.path.normpath('../datasets')

face_predictor_path = os.path.normpath('../common/predictors/shape_predictor_68_face_landmarks.dat')

absolute_max_string_len = 32