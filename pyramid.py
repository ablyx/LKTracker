import numpy as np
from LKTracker import LKTracker
from sampling import down_sample

DOWN_RES_LIMIT = 64
## Given two images, track the features from one image to the next using LK Tracker and Pyramid

def down_sample_feature_coords(feature):
    return [feature[0] / 2, feature[1] / 2]

def up_sample_feature_coords(feature):
    return [feature[0]*2, feature[1]*2]

def get_feature_coords_for_next_frame(prev_frame, next_frame, prev_features):
    prev_frame = frame.astype('int16')
    next_frame = next_frame.astype('int16')
    frame_res_row = prev_frame.shape[0]
    # Downsample then LK when recurse up
    # Base case
    if frame_res_row <= DOWN_RES_LIMIT:
        # Run LK Tracker and return result
        return LKTracker(prev_frame, next_frame, prev_features, prev_features)

    # Recurse
    prev_frame_small = down_sample(prev_frame)
    next_frame_small = down_sample(next_frame)
    downsampled_features = [down_sample_feature_coords(feature) for feature in prev_features]

    # LK Track based on features from previous recursion call
    features_after_delta = get_feature_coords(prev_frame_small, next_frame_small, downsampled_features)
    upsampled_features_after_delta = [up_sample_feature_coords(feature) for feature in upsampled_features_after_delta]

    return LKTracker(prev_frame, next_frame, prev_features, upsampled_features_after_delta)
