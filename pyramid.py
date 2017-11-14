import numpy as np
import LKTracker as lkt
from sampling import down_sample

## Given two images, track the features from one image to the next using LK Tracker and Pyramid

def get_feature_coords(prev_frame, next_frame, prev_features):
    prev_frame = frame.astype('int16')
    next_frame = next_frame.astype('int16')
    prev_frame.shape()

    # down_sample()
    # prev_feature
