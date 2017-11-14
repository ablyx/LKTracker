import copy
import cv2
import numpy as np
import pickle
from scipy.signal import convolve2d
from utils import *
from sampling import down_sample

# cv2.cv.CV_CAP_PROP_FPS
CV_CAP_PROP_FPS = 5
NUM_FEATURES = 20
DOWN_RES_LIMIT = 2*64

## Given two images, track the features from one image to the next using LK Tracker and Pyramid
def down_sample_feature_coords(feature):
    return [feature[0]/2, feature[1]/2]

def up_sample_feature_coords(feature):
    return [feature[0]*2, feature[1]*2]

def get_feature_coords_for_next_frame(prev_frame, next_frame, prev_features):
    prev_frame = prev_frame.astype('int16')
    next_frame = next_frame.astype('int16')
    frame_res_row = prev_frame.shape[0]
    frame_res_col = prev_frame.shape[1]
    print("Frame res row: ", frame_res_row, "col: ", frame_res_col)
    # Downsample to base case then "upsample" and LK when recursing up
    # Base case
    if frame_res_row <= DOWN_RES_LIMIT:
        # Run LK Tracker and return result
        return LKTracker(prev_frame, next_frame, prev_features, prev_features)

    # Recurse
    prev_frame_small = down_sample(prev_frame)
    next_frame_small = down_sample(next_frame)
    downsampled_features = [down_sample_feature_coords(feature) for feature in prev_features]
    print('Downsampled')
    print(prev_frame_small.shape)
    # LK Track based on features from previous recursion call
    features_after_delta = get_feature_coords_for_next_frame(prev_frame_small, next_frame_small, downsampled_features)
    upsampled_features_after_delta = [up_sample_feature_coords(feature) for feature in features_after_delta]

    return LKTracker(prev_frame, next_frame, prev_features, upsampled_features_after_delta)


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def get_features(frame0):
    with open('features.pickle', 'rb') as pkl:
        features = pickle.load(pkl)
    return features
    features = []
    frame = frame0.astype('int16')
    rows, cols = frame.shape[:2]

    # getting the derivatives
    # Ix and Iy store the gx and gy (derivatives) values
    # Ixx, Ixy and Iyy store the product of gx and gy
    gx = copy.deepcopy(frame)
    gy = copy.deepcopy(frame)
    print('get gradient')
    for row in range(rows - 1):
        for col in range(cols):
            gx[row][col] = int(frame[row + 1][col]) - int(frame[row][col])

    for row in range(rows):
        for col in range(cols - 1):
            gy[row][col] = int(frame[row][col + 1]) - int(frame[row][col])

    Ixx = np.multiply(gx, gx)
    Ixy = np.multiply(gx, gy)
    Iyy = np.multiply(gy, gy)

    # get Zd = b
    # using a window of all ones instead of gaussian for speed (and convenience)
    window = np.ones((13, 13))
    # convolve gradients with window of interest
    # these W matrices are used to get Z
    Wxx = convolve2d(Ixx, window, mode='full')
    Wxy = convolve2d(Ixy, window, mode='full')
    Wyy = convolve2d(Iyy, window, mode='full')

    # each pixel in the image has a corresponding window which gives matrix Z
    # getting the smallest eigenvalue of Z
    print('get eig')
    wrows, wcols = Wyy.shape
    eigvals = np.zeros((wrows, wcols))
    for row in range(wrows):
        for col in range(wcols):
            W = np.matrix([[Wxx[row][col], Wxy[row][col]], [Wxy[row][col], Wyy[row][col]]])
            vals, vec = np.linalg.eig(W)
            eigvals[row][col] = min(vals)

    # split the image into 13 by 13 windows
    # for each window, take the highest eigenvalue and set the rest to 0
    # identify which pixels are good features
    print('get mosaic eig')

    for row in range(0, wrows - 13, 13):
        for col in range(0, wcols - 13, 13):
            window = eigvals[row:row + 13, col:col + 13]
            largest_eig = max(window.ravel())
            window = map(lambda row: map(lambda x: 0 if x != largest_eig else x, row), window)
            eigvals[row:row + 13, col:col + 13] = window

    print('track done, plot corners')
    # plot the 200 best features
    ex, ey = largest_indices(eigvals, NUM_FEATURES)
    for i, x in enumerate(ex):
        y = ey[i]
        features.append((y - 7, x - 7))
    return features


def test_get_features():
    frame = cv2.imread('test.jpg', 0)
    features = get_features(frame)
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    print(features)
    for y,x in features:
        cv2.circle(color_frame, (y, x), 1, (0, 0, 255), -1)
    cv2.imwrite('FEATURES200.jpg', color_frame)
# test_get_features()


# frames are grayscale and of the same size
# should return centers of tracked features for better(larger) resolution of next_frame
# returns the next next_frame_features

#  Outputs the feature coords of the next frame, at the larger resolution
def LKTracker(frame, next_frame, frame_features, next_frame_features):
    frame = frame.astype('int16')
    next_frame = next_frame.astype('int16')
    rows, cols = frame.shape[:2]
    # print(rows, cols)


    # getting the derivatives
    # Ix and Iy store the gx and gy (derivatives) values
    # Ixx, Ixy and Iyy store the product of gx and gy
    gx = copy.deepcopy(frame)
    gy = copy.deepcopy(frame)
    print('get gradient')
    for row in range(rows-1):
        for col in range(cols):
            gx[row][col] = int(frame[row+1][col]) - int(frame[row][col])

    for row in range(rows):
        for col in range(cols-1):
            gy[row][col] = int(frame[row][col+1]) - int(frame[row][col])

    Ixx = np.multiply(gx, gx)
    Ixy = np.multiply(gx, gy)
    Iyy = np.multiply(gy, gy)

    # get Zd = b

    # getting b
    I = copy.deepcopy(frame)
    J = copy.deepcopy(next_frame)

    larger_reso_next_frame_features = []
    # feature is in the form (y,x)
    for i, prev_feature in enumerate(frame_features):
        prev_y, prev_x = prev_feature
        Wxx = sum(Ixx[prev_y - 6: prev_y + 7, prev_x - 6: prev_x + 7].ravel())
        Wxy = sum(Ixy[prev_y - 6: prev_y + 7, prev_x - 6: prev_x + 7].ravel())
        Wyy = sum(Iyy[prev_y - 6: prev_y + 7, prev_x - 6: prev_x + 7].ravel())

        Z = np.matrix([[Wxx, Wxy], [Wxy, Wyy]])

        prev_feature_window = I[prev_y-6: prev_y+7, prev_x-6: prev_x+7]
        next_y, next_x = next_frame_features[i]
        next_feature_window = J[next_y - 6: next_y + 7, next_x - 6: next_x + 7]
        # skip the boundary cases
        if prev_feature_window.shape != (13,13) or next_feature_window.shape != (13,13): 
            larger_reso_next_frame_features.append((next_y+0, next_x+0))
            continue
        # w(I-J)
        window_diff = prev_feature_window - next_feature_window
        bx = sum(np.multiply(window_diff, gx[prev_y-6: prev_y+7, prev_x-6: prev_x+7]).ravel())
        by = sum(np.multiply(window_diff, gy[prev_y-6: prev_y+7, prev_x-6: prev_x+7]).ravel())
        try:
            # solve for d
            Z_inv = np.linalg.inv(Z)
            b = np.matrix([[bx], [by]])
            dy, dx = np.dot(Z_inv, b)
            dx = int(dx)
            dy = int(dy)
            # print("Success")
        except:
            print('error solving Zd = b')
            dx, dy = 0, 0

        # Next frame features at higher res, but have not upsampled yet
        larger_reso_next_frame_features.append((next_y+dy, next_x+dx))
    return larger_reso_next_frame_features

def test_LKTracker():
    frame0 = cv2.imread('lk_test0.jpg', 0)
    frame1 = cv2.imread('lk_test1.jpg', 0)
    with open('features.pickle', 'rb') as pkl:
        features = pickle.load(pkl)
    # features = get_features(frame0)
    # with open('features.pickle', 'wb') as output:
    #     pickle.dump(features, output, pickle.HIGHEST_PROTOCOL)
    next_features = get_feature_coords_for_next_frame(frame0, frame1, features)
    print(next_features)
    # for y,x in features:
    #     cv2.circle(color_frame, (y, x), 1, (0, 0, 255), -1)
    # cv2.imwrite('FEATURES200.jpg', color_frame)
# test_LKTracker()

# lets see if my code works
# get the first 2 frames
cap = cv2.VideoCapture('test/clip2.mp4')

fps = cap.get(CV_CAP_PROP_FPS)
print(fps)

img_arr = []

# First frame
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame1_features = get_features(frame)
print("Frame1 features:", frame1_features)
color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
for y,x in frame1_features:
    cv2.circle(color_frame, (y, x), 1, (0, 0, 255), -1)
# cv2.imwrite('lk_test1.jpg', color_frame)
img_arr.append(color_frame)


## Looping to get the rest of the frames from video
prev_features = frame1_features
prev_frame = frame
# Get remaining frames
frame_counter = 0
while True:
    # Get frame
    ret, frame = cap.read()

    if frame is None:
        break
    # Convert frame to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_features = prev_features

    # Get LKTracker feature coordinates
    # Pyramid is not working yet
    result = get_feature_coords_for_next_frame(prev_frame, frame, prev_features)

    # result = LKTracker(prev_frame, frame, prev_features, frame_features)

    ## Draw circles of LKTracker result on this frame
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for y,x in result:
        cv2.circle(color_frame, (y, x), 1, (0, 0, 255), -1)

    if frame_counter < 2:
        f_name = 'lk_test' + str(frame_counter) + '.jpg'
        cv2.imwrite(f_name, color_frame)

    img_arr.append(color_frame)

    # Update prev_features and prev_frame
    prev_features = result
    prev_frame = frame
    frame_counter += 1
    print(frame_counter)

with open('vid.pickle', 'wb') as output:
    pickle.dump(img_arr, output, pickle.HIGHEST_PROTOCOL)

write_img_array_to_video(img_arr, fps, 'lk_test_vid.avi')


# eigvals are gotten from the first frame
# correspond to good features
# cv2.imwrite('test.jpg', frame)

## Testing LKTracker
##



# dx, dy, eigvals = LKTracker(frame, next_frame)
# print('track done, plot corners')
# # plot the 200 best features
# ex, ey = largest_indices(eigvals, 200)
# color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
# for i, x in enumerate(ex):
#     y = ey[i]
#     cv2.circle(color_frame, (y-7,x-7), 1, (0,0,255), -1)
#
# # cv2.imshow('color frame', color_frame)
# cv2.imwrite('corners200.jpg', color_frame)
# # cv2.waitKey(0)

