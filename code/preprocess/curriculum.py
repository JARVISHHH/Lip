import numpy as np
from preprocess.video import VideoAugmenter
import hyperparameters as hp
import tensorflow as tf

class Curriculum(object):
    def __init__(self, rules):
        self.rules = rules
        self.train = False
        self.epoch = -1
        self.sentence_length = -1
        self.flip_probability = 0.0
        self.jitter_probability = 0.0
        self.gaussian_probability = 0.0
        self.decay_rate = 1.0

    def update(self, epoch, train=True):
        self.epoch = epoch
        self.train = train
        current_rule = self.rules(self.epoch)
        self.sentence_length = current_rule.get('sentence_length') or -1
        self.flip_probability = current_rule.get('flip_probability') or 0.0
        self.jitter_probability = current_rule.get('jitter_probability') or 0.0
        self.gaussian_probability = current_rule.get('gaussian_probability') or 0.0
        self.decay_rate = current_rule.get('decay_rate') or 1.0
        

    def apply(self, video, align):
        original_video = video
        if self.sentence_length > 0:
            video, align = VideoAugmenter.pick_subsentence(video, align, self.sentence_length)
        # Only apply horizontal flip and temporal jitter on training
        if self.train:
            if np.random.ranf() < self.flip_probability:
                video = VideoAugmenter.horizontal_flip(video)
            # added random gaussian_noise
            if np.random.ranf() < self.gaussian_probability:
                video = VideoAugmenter.add_gaussian_noise(video, mean=0, std=0.01)

            video = VideoAugmenter.temporal_jitter(video, self.jitter_probability)
            
            # Control additional word-level training instances based on decay rate
            if np.random.ranf() < self.decay_rate:
                video, align = VideoAugmenter.pick_word(video, align)
        video_unpadded_length = video.length
        video = VideoAugmenter.pad(video, original_video.length)
        return video, align, video_unpadded_length

    def __str__(self):
        return "{}(train: {}, sentence_length: {}, flip_probability: {}, jitter_probability: {}, gaussian_probability: {}, decay_rate: {})"\
            .format(self.__class__.__name__, self.train, self.sentence_length, self.flip_probability, self.jitter_probability, self.gaussian_probability, self.decay_rate)
            
class CurriculumUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, curriculum):
        super(CurriculumUpdateCallback, self).__init__()
        self.curriculum = curriculum
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.curriculum:
            self.curriculum.update(epoch, train=True)  
            print(f"Curriculum updated for epoch {epoch} : {self.curriculum}")
