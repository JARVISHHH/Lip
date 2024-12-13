import numpy as np
import tensorflow as tf

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
    
    # def get_padded_label(self, label):
    #     padding = np.zeros((hp.absolute_max_string_len - len(label)))
    #     return np.concatenate((np.array(label), padding), axis=0)
    
    
    def build(self, align):
        self.align = self.strip(align, ['sp', 'sil'])
        self.sentence = self.get_sentence(align)
        self.tokens = []
        for c in self.align:
            self.tokens = [*self.tokens, ' ', c[-1]]
        self.tokens = self.tokens[1:]
        self.label = self.get_label(tf.reshape(tf.strings.unicode_split(self.tokens, input_encoding='UTF-8'), (-1)))
        self.padded_label = tf.pad(
            self.label,
            [[0, hp.absolute_max_string_len - tf.shape(self.label)[0]]]
        )

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(y[0], y[1], y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self
    
    def from_array(self, align):
        self.build(align)
        return self
    '''
    def parse_alignment(self, align):
        """
        Convert start and end times in the alignment to integers.
        Args:
            align (list): List of alignment tuples (start, end, token).

        Returns:
            list: Updated alignment with start and end as integers.
        """
        return [(int(sub[0]), int(sub[1]), sub[2]) for sub in align]

    def process_labels(self, align):
        """
        Process alignments to produce labels and padded labels.

        Args:
            align (list): List of alignment tuples.

        Returns:
            tuple: (labels, padded_labels).
        """
        stripped_align = self.strip(align, ['sp', 'sil'])
        sentence = self.get_sentence(stripped_align)
        
        # Debug: Check for unknown tokens
        try:
            labels = self.get_label(sentence)
        except tf.errors.InvalidArgumentError as e:
            print(f"Error: Token not in vocabulary. Sentence: {sentence}")
            raise e
        padding = np.ones((hp.absolute_max_string_len - len(labels))) * -1
        padded_labels = np.concatenate((np.array(labels), padding), axis=0)
        return labels, padded_labels
    '''
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
