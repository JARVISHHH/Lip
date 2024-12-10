import numpy as np
from video import VideoAugmenter
import hyperparameters as hp

#add to train.py later
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

class Curriculum(object):
    def __init__(self, rules):
        self.rules = rules
        self.epoch = -1

    def update(self, epoch, train=True):
        self.epoch = epoch
        self.train = train
        current_rule = self.rules(self.epoch)
        self.sentence_length = current_rule.get('sentence_length') or -1
        self.flip_probability = current_rule.get('flip_probability') or 0.0
        self.jitter_probability = current_rule.get('jitter_probability') or 0.0
        self.noise_probability = current_rule.get('noise_probability') or 0.0
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
            if np.random.ranf() < self.noise_probability:
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
