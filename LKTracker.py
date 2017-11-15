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
WINDOW = 13
WW = int(round(WINDOW/2))

## Given two images, track the features from one image to the next using LK Tracker and Pyramid
def down_sample_feature_coords(feature):
    # got chance for error?
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
    # with open('features.pickle', 'rb') as pkl:
    #     features = pickle.load(pkl)
    # return features
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
    window = np.ones((WINDOW, WINDOW))
    # convolve gradients with window of interest
    # these W matrices are used to get Z
    Wxx = convolve2d(Ixx, window, mode='full')
    Wxy = convolve2d(Ixy, window, mode='full')
    Wyy = convolve2d(Iyy, window, mode='full')

    # each pixel in the image has a corresponding window which gives matrix Z
    # getting the smallest eigenvalue of Z
    
    print('get eig')
    wrows, wcols = Wyy.shape
    Zs = np.zeros((wrows, wcols, 2,2))
    eigvals = np.zeros((wrows, wcols))
    for row in range(wrows):
        for col in range(wcols):
            W = np.matrix([[Wxx[row][col], Wxy[row][col]], [Wxy[row][col], Wyy[row][col]]])
            Zs[row][col] = W
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
    gdZs = []
    ey, ex = largest_indices(eigvals, NUM_FEATURES)
    for i, x in enumerate(ex):
        y = ey[i]
        # this is correct. the Z matches for nonconv if nonconv just takes in y,x
        # features.append((y-WW, x-WW))
        gdZs.append(Zs[y][x])
        # this is for conv
        features.append((y, x))
    return features, gdZs


def test_get_features():
    frame = cv2.imread('test.jpg', 0)
    features = get_features(frame)
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    print(features)
    for y,x in features:
        cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
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
    # using a window of all ones instead of gaussian for speed (and convenience)
    window = np.ones((WINDOW, WINDOW))

    # getting b
    I = copy.deepcopy(frame)
    J = copy.deepcopy(next_frame)

    result_features = []
    Zs = np.zeros((NUM_FEATURES, 2, 2))
    # feature is in the form (y,x)
    # print(frame_features)
    for i, prev_feature in enumerate(frame_features):
        prev_y, prev_x = prev_feature
        # TAKE NOTE OF NEGATIVE/out of bounds INDEXES
        x_start = max(0, prev_x - WW)
        x_end = min(cols, prev_x + WW + 1)
        y_start = max(0, prev_y - WW)
        y_end = min(rows, prev_y + WW + 1)
        # print(x_start)
        # print(x_end)
        # print(y_start)
        # print(y_end)
        Wxx = sum(Ixx[y_start: y_end, x_start: x_end].ravel())
        Wxy = sum(Ixy[y_start: y_end, x_start: x_end].ravel())
        Wyy = sum(Iyy[y_start: y_end, x_start: x_end].ravel())

        Z = np.matrix([[Wxx, Wxy], [Wxy, Wyy]])
        Zs[i] = Z
        prev_feature_window = I[y_start: y_end, x_start: x_end]
        next_y, next_x = next_frame_features[i]
        next_x_start = max(0, next_x - WW)
        next_x_end = min(cols, next_x + WW + 1)
        next_y_start = max(0, next_y - WW)
        next_y_end = min(rows, next_y + WW + 1)
        # print(next_x_start)
        # print(next_x_end)
        # print(next_y_start)
        # print(next_y_end)
        next_feature_window = J[next_y_start: next_y_end, next_x_start: next_x_end]
        # # skip the boundary cases
        if prev_feature_window.shape != (WINDOW,WINDOW) or next_feature_window.shape != (WINDOW,WINDOW): 
            result_features.append((next_y+0, next_x+0))
            continue
        ## w(I-J)
        window_diff = prev_feature_window - next_feature_window
        bx = sum(np.multiply(window_diff, gx[y_start: y_end, x_start: x_end]).ravel())
        by = sum(np.multiply(window_diff, gy[y_start: y_end, x_start: x_end]).ravel())

        try:
            # solve for d
            # print(Z)
            Z_inv = np.linalg.inv(Z)
            # print('inverse')
            b = np.matrix([[bx], [by]])
            # print(b)
            d = np.dot(Z_inv, b)
            # print('dot')
            dx, dy = d
            dx = int(round(dx))
            dy = int(round(dy))
            ny, nx = (next_y-dy, next_x-dx)
            # print(d)
            if (dx,dy)!=(0,0):
                print(dx,dy)
        except:
            print('error solving Zd = b')
            dx, dy = 0, 0
            ny, nx = 0, 0

        # Next frame features at higher res, but have not upsampled yet
        # next frame features shld nt be negative
        # set to (0,0) if negative
        result_features.append((ny, nx))
    return result_features, Zs

#  Outputs the feature coords of the next frame, only small movments
def LKTrackerConv(frame, next_frame, frame_features, next_frame_features):
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
    # using a window of all ones instead of gaussian for speed (and convenience)
    window = np.ones((WINDOW, WINDOW))
    # convolve gradients with window of interest
    # these W matrices are used to get Z
    Wxx = convolve2d(Ixx, window, mode='full')
    Wxy = convolve2d(Ixy, window, mode='full')
    Wyy = convolve2d(Iyy, window, mode='full')

    # getting b
    I = copy.deepcopy(frame)
    J = copy.deepcopy(next_frame)
    IminusJgx = convolve2d(np.multiply(I-J, gx), window, mode='full')
    IminusJgy = convolve2d(np.multiply(I-J, gy), window, mode='full')

    next_frame_features = []
    Zs = np.zeros((NUM_FEATURES, 2, 2))
    # feature is in the form (y,x)
    # print(frame_features)
    for i, prev_feature in enumerate(frame_features):
        prev_y, prev_x = prev_feature
        # if prev_y == 0 and prev_x == 0:
        #     # skip (0,0) case. exception. screwed up somewhere.
        #     larger_reso_next_frame_features.append((0, 0))
        # TAKE NOTE OF NEGATIVE INDEXES
        Z = np.matrix([[Wxx[prev_y][prev_x], Wxy[prev_y][prev_x]], [Wxy[prev_y][prev_x], Wyy[prev_y][prev_x]]])
        Zs[i] = Z
        bx = IminusJgx[prev_y][prev_x]
        by = IminusJgy[prev_y][prev_x]
        try:
            # vals, vec = np.linalg.eig(Z)
            # eigval = min(vals)
            # solve for d
            Z_inv = np.linalg.inv(Z)
            b = np.matrix([[bx], [by]])
            d = np.dot(Z_inv, b)
            dx, dy = d
            dx = int(round(dx))
            dy = int(round(dy))
            if (dx,dy)!=(0,0):
                print(d)
                print(dx,dy)
        except:
            print('error solving Zd = b')
            dx, dy, e = 0, 0, 0

        # Next frame features at higher res, but have not upsampled yet
        # next frame features shld nt be negative
        # set to (0,0) if negative
        next_frame_features.append((prev_y-dy, prev_x-dx))
    return next_frame_features, Zs

def test_LKTracker():
    frame0 = cv2.imread('test/t1.jpeg', 0)
    frame1 = cv2.imread('test/t4.jpeg', 0)
    
    # with open('features.pickle', 'rb') as pkl:
    #     features = pickle.load(pkl)
    features, z = get_features(frame0)
    print(features)
    # print(z)
    color_frame = cv2.cvtColor(frame0, cv2.COLOR_GRAY2RGB)
    for y,x in features:
        cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite('im1features.jpg', color_frame)
    # with open('features.pickle', 'wb') as output:
    #     pickle.dump(features, output, pickle.HIGHEST_PROTOCOL)
    next_features,z = LKTrackerConv(frame0, frame1, features, features)
    print(next_features)
    # print(z)
    color_frame = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
    for y,x in next_features:
        cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite('im2features.jpg', color_frame)
    # features = get_features(frame1)
    # print(features)
test_LKTracker()
"""
# lets see if my code works
# get the first 2 frames
cap = cv2.VideoCapture('test/clip.mp4')

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
    cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
# cv2.imwrite('lk_test1.jpg', color_frame)
cv2.imwrite('lk_test0.jpg', color_frame)
img_arr.append(color_frame)


## Looping to get the rest of the frames from video
prev_features = frame1_features
prev_frame = frame
# Get remaining frames
frame_counter = 1
# while frame_counter<50:
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
    # result = get_feature_coords_for_next_frame(prev_frame, frame, prev_features)

    result = LKTracker(prev_frame, frame, prev_features, frame_features)
    # result = LKTrackerConv(prev_frame, frame, prev_features, frame_features)

    ## Draw circles of LKTracker result on this frame
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for y,x in result:
        cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)

    if frame_counter < 2:
        f_name = 'lk_test' + str(frame_counter) + '.jpg'
        cv2.imwrite(f_name, color_frame)

    img_arr.append(color_frame)

    # Update prev_features and prev_frame
    prev_features = result
    prev_frame = frame
    frame_counter += 1
    print(frame_counter)

# with open('vidnopyr.pickle', 'wb') as output:
#     pickle.dump(img_arr, output, pickle.HIGHEST_PROTOCOL)

write_img_array_to_video(img_arr, fps, 'lk_clip_noconv.avi')


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
"""
