import os
# import skvideo.io
import tensorflow as tf
import numpy as np
import cv2
import dlib

import hyperparameters as hp

class Video(object):
    def __init__(self, video_type = 'face'):
        self.video_type = video_type

    def from_video(self, path):
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self
    
    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [cv2.resize(cv2.imread(frame_path), np.array([hp.image_width, hp.image_height])) for frame_path in frames_path]
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
        self.set_data(mouth_frames)

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

        middle_frame = frames[int(len(frames) / 3)]
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