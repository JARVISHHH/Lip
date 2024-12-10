import os
# import skvideo.io
import tensorflow as tf
import numpy as np
import cv2
import dlib
import hyperparameters as hp
from .align import Align

class VideoAugmenter(object):
    @staticmethod
    def split_words(video, align):
        video_aligns = []
        for sub in align.align:
            # Create new video
            _video = Video(video.video_type)
            _video.face = video.face[sub[0]:sub[1]]
            _video.mouth = video.mouth[sub[0]:sub[1]]
            _video.set_data(_video.mouth)
            # Create new align
            _align = Align(align.absolute_max_string_len, align.label_func).from_array([(0, sub[1]-sub[0], sub[2])])
            # Append
            video_aligns.append((_video, _align))
        return video_aligns

    @staticmethod
    def merge(video_aligns):
        vsample = video_aligns[0][0]
        asample = video_aligns[0][1]
        video = Video(vsample.video_type)
        video.face = np.ones((0, vsample.face.shape[1], vsample.face.shape[2], vsample.face.shape[3]), dtype=np.uint8)
        video.mouth = np.ones((0, vsample.mouth.shape[1], vsample.mouth.shape[2], vsample.mouth.shape[3]), dtype=np.uint8)
        align = []
        inc = 0
        for _video, _align in video_aligns:
            video.face = np.concatenate((video.face, _video.face), 0)
            video.mouth = np.concatenate((video.mouth, _video.mouth), 0)
            for sub in _align.align:
                _sub = (sub[0]+inc, sub[1]+inc, sub[2])
                align.append(_sub)
            inc = align[-1][1]
        video.set_data(video.mouth)
        align = Align(asample.absolute_max_string_len, asample.label_func).from_array(align)
        return (video, align)

    @staticmethod
    def pick_subsentence(video, align, length):
        split = VideoAugmenter.split_words(video, align)
        start = np.random.randint(0, align.word_length - length)
        return VideoAugmenter.merge(split[start:start+length])

    @staticmethod
    def pick_word(video, align):
        video_aligns = np.array(VideoAugmenter.split_words(video, align))
        return video_aligns[np.random.randint(video_aligns.shape[0], size=2), :][0]

    @staticmethod
    def horizontal_flip(video):
        _video = Video(video.video_type)
        _video.face = np.flip(video.face, 2)
        _video.mouth = np.flip(video.mouth, 2)
        _video.set_data(_video.mouth)
        return _video

    @staticmethod
    def temporal_jitter(video, probability):
        changes = [] # [(frame_i, type=del/dup)]
        t = video.length
        for i in range(t):
            if np.random.ranf() <= probability/2:
                changes.append((i, 'del'))
            if probability/2 < np.random.ranf() <= probability:
                changes.append((i, 'dup'))
        _face = np.copy(video.face)
        _mouth = np.copy(video.mouth)
        j = 0
        for change in changes:
            _change = change[0] + j
            if change[1] == 'dup':
                _face = np.insert(_face, _change, _face[_change], 0)
                _mouth = np.insert(_mouth, _change, _mouth[_change], 0)
                j = j + 1
            else:
                _face = np.delete(_face, _change, 0)
                _mouth = np.delete(_mouth, _change, 0)
                j = j - 1
        _video = Video(video.video_type)
        _video.face = _face
        _video.mouth = _mouth
        _video.set_data(_video.mouth)
        return _video

    @staticmethod
    def pad(video, length):
        pad_length = max(length - video.length, 0)
        video_length = min(length, video.length)
        face_padding = np.ones((pad_length, video.face.shape[1], video.face.shape[2], video.face.shape[3]), dtype=np.uint8) * 0
        mouth_padding = np.ones((pad_length, video.mouth.shape[1], video.mouth.shape[2], video.mouth.shape[3]), dtype=np.uint8) * 0
        _video = Video(video.video_type)
        _video.face = np.concatenate((video.face[0:video_length], face_padding), 0)
        _video.mouth = np.concatenate((video.mouth[0:video_length], mouth_padding), 0)
        _video.set_data(_video.mouth)
        return _video   

    @staticmethod
    def standardize(video):
        _video = Video(video.video_type)
        _video.face = (video.face - np.mean(video.face, axis=(1, 2, 3), keepdims=True) / 
            (np.std(video.face, axis=(1, 2, 3), keepdims=True) + 1e-8))
        _video.mouth = (video.mouth - np.mean(video.mouth, axis=(1, 2, 3), keepdims=True) / 
            (np.std(video.mouth, axis=(1, 2, 3), keepdims=True) + 1e-8))
        _video.set_data(_video.mouth)
        return _video

    @staticmethod
    def add_gaussian_noise(video, mean=0, std=0.01):
        _video = Video(video.video_type)
        noise_face = np.random.normal(mean, std, video.face.shape)
        noise_mouth = np.random.normal(mean, std, video.mouth.shape)

        _video.face = np.clip(video.face + noise_face, 0, 255).astype(np.uint8)
        _video.mouth = np.clip(video.mouth + noise_mouth, 0, 255).astype(np.uint8)
        _video.set_data(_video.mouth)
        return _video

        
class Video(object):
    def __init__(self, video_type = 'face'):
        self.video_type = video_type

    def from_video(self, path):
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self
    
    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [cv2.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self
    
    def handle_type(self, frames):
        if self.video_type == 'mouth':
            self.process_frames_mouth(frames)
        elif self.video_type == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    def get_video_frames(self, path: str):
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
            _, frame = cap.read()
            frames.append(frame)
        cap.release()
        return frames
    
    def process_frames_face(self, frames):
        self.face = np.array(frames)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(hp.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)
    
    def get_frames_mouth(self, detector, predictor, frames):
        MOUTH_WIDTH = hp.image_width
        MOUTH_HEIGHT = hp.image_height
        HORIZONTAL_PAD = 0.19
        mouth_left = None
        mouth_frames = []

        middle_frame = frames[int(hp.frames_number / 3)]
        try:
            dets = detector(middle_frame, 1)
            shape = None
            for _, d in enumerate(dets):
                shape = predictor(middle_frame, d)
            if shape is None:  # Detector doesn't detect face, just return as is
                return middle_frame
            mouth_points = []
            i = -1
            
            for part in shape.parts():
                i += 1
                if i < 48 or i > 68:  # Only take mouth region
                    continue
                mouth_points.append((part.x, part.y))
            np_mouth_points = np.array(mouth_points)

            if mouth_left is None:
                mouth_left = int(np.min(np_mouth_points[:, 0]) * (1.0 - HORIZONTAL_PAD))
                mouth_right = int(np.max(np_mouth_points[:, 0]) * (1.0 + HORIZONTAL_PAD))

                mouth_top = int(np.min(np_mouth_points[:, 1]) * (1.0 - HORIZONTAL_PAD))
                mouth_bottom = int(np.max(np_mouth_points[:, 1]) * (1.0 + HORIZONTAL_PAD))
        except:
            return middle_frame

        for frame in frames:
            try:
                dets = detector(frame, 1)
                shape = None
                for _, d in enumerate(dets):
                    shape = predictor(frame, d)
                if shape is None:  # Detector doesn't detect face, just return as is
                    return frame
                mouth_points = []
                i = -1
                
                for part in shape.parts():
                    i += 1
                    if i < 48 or i > 68:  # Only take mouth region
                        continue
                    mouth_points.append((part.x, part.y))
                np_mouth_points = np.array(mouth_points)

                mouth_crop_image = cv2.resize(frame[mouth_top:mouth_bottom, mouth_left:mouth_right], np.array([MOUTH_WIDTH, MOUTH_HEIGHT]))

                mouth_frames.append(mouth_crop_image)
            except:
                return frame

        return mouth_frames

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            try:
                frame = frame.swapaxes(0,1)  # swap width and height to form format W x H x C
                if len(frame.shape) < 3:
                    frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1)  # Add grayscale channel
                data_frames.append(frame)
            except:
                pass
        self.data = np.array(data_frames)
        self.length = len(data_frames)