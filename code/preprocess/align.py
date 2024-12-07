import numpy as np

import hyperparameters as hp

class Align(object):
    def __init__(self, label_func=None):
        self.label_func = label_func

    def strip(self, align, items):
        """
        strip remove items from align

        :param align: original align list
        :param items: items to be removed
        """
        return [sub for sub in align if sub[2] not in items]
    
    def get_sentence(self, align):
        """
        get_sentence concatenate align into a string sentence (sp and sil are removed)

        :param align: original align list
        """
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])
    
    def get_label(self, sentence):
        return self.label_func(sentence)
    
    def get_padded_label(self, label):
        padding = np.zeros((hp.absolute_max_string_len - len(label)))
        return np.concatenate((np.array(label), padding), axis=0)
    
    def build(self, align):
        self.align = self.strip(align, ['sp', 'sil'])
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(y[0], y[1], y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self
    
    def from_array(self, align):
        self.build(align)
        return self

    @property
    def word_length(self):
        """
        the number of words in the sentence
        """
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        """
        the number of letters (including spaces) in the sentence
        """
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)
