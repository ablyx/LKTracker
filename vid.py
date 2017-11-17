from homography import new_warp_subimage
from utils import *
"""
video = None
firstFrame = None
prevFrame = firstFrame.deepcopy()
#clockwise from top left corner
# hardcoded pts from first image. might not be rectangle
orig_pts = [p1, p2, p3, p4]
xcoords = map(lambda p:p[1], orig_pts)
ycoords = map(lambda p:p[0], orig_pts)
p1 = (min(xcoords), min(ycoords))
p2 = (max(xcoords), min(ycoords))
p3 = (min(xcoords), max(ycoords))
p4 = (max(xcoords), max(ycoords))
# this is a rectangle based on the pts from first image
vortex_pts = [p1, p2, p3, p4]
vortex = cv2.imread('vortex.jpg')
vortex = cv2.resize(vortex, (p4 - p1)[1], (p2-p1)[0]) # y,x
firstFrameVortex = warp_subimage(firstFrame, vortex, vortex_pts, orig_pts)
img_arr = [firstFrameVortex,]

for frame in video[1:]:
    to_warp_pts = track(prev_frame, frame, [p1, p2, p3, p4])
    prev_frame = frame.deepcopy()
    frameVortex = warp_subimage(frame, vortex, vortex_pts, to_warp_pts)
    img_arr.append(frameVortex)
"""

import numpy as np
import cv2
cap = cv2.VideoCapture('vortex.mp4')
fps = cap.get(5)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (13,13),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print('p0', p0)
# rectangle
# in the form x, y
orig_pts = [(286,129),(282,329),(535,135),(529,329)]
# in the form x, y
# orig_pts = [(129,286),(329,282),(135,535),(329,529)]
p0 = np.array(map(lambda p: np.array(p), orig_pts))
p0 = p0.astype('float32')
# p0 = np.array(orig_pts)

xcoords = map(lambda p:p[0], orig_pts)
ycoords = map(lambda p:p[1], orig_pts)
p1 = np.array((min(xcoords), min(ycoords)))
p2 = np.array((min(xcoords), max(ycoords)))
p3 = np.array((max(xcoords), min(ycoords)))
p4 = np.array((max(xcoords), max(ycoords)))

# this is a rectangle based on the pts from first image
vortex_pts = [p1, p2, p3, p4]
print('orig_pts', orig_pts)
print('vortex_pts', vortex_pts)
rows = (p4-p1)[0]
cols = (p4-p1)[1]
vortex_corner_pts = [(0, 0), (0, rows-1), (cols-1, 0), (cols-1, rows-1)]
vortex_corner_pts = map(lambda p: np.array(p), vortex_corner_pts)
p1, p2, p3, p4 = vortex_corner_pts
# print(vortex_pts)
vortexes = []
for i in range(1,13):
    vortex = cv2.imread('vortex/v{}.tiff'.format(i))
    vortex = cv2.resize(vortex, (((p4-p1)[1]), (p4 - p1)[0]))
    vortexes.append(vortex)
# should i invert vortex
# vortex = (255-vortex)
# cv2.imshow('test2',vortex)
print( ((p4 - p1)[1], (p4-p1)[0]))
vortex = cv2.resize(vortex, (((p4-p1)[1]), (p4 - p1)[0])) # y,x
firstFrame = old_frame
firstFrameVortex = new_warp_subimage(firstFrame, vortexes[-1], orig_pts)
# cv2.namedWindow("vortex", cv2.WINDOW_NORMAL)
img_arr = [firstFrameVortex,]
# cv2.imshow('vortex',firstFrameVortex)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
counter = 1

while(True):
    ret,frame = cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    # print('p0',p0)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # board_pts = list(map(lambda p:p, p1))
    board_pts = list(p1)
    # print('good new', good_new)

    # print('gn', good_new[0])
    # print('good new', list(map(lambda p:list(p), good_new[0])))
    # draw the tracks
    v = counter % 12
    vortex = vortexes[v]
    frameVortex = new_warp_subimage(frame, vortex, board_pts)
    img_arr.append(frameVortex)
    # for i,(new,old) in enumerate(zip(good_new,good_old)):

        # a,b = new.ravel()
        # c,d = old.ravel()
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    # cv2.imshow('vortex',frameVortex)
    # cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = p1
    counter = counter + 1
    print(counter)

# cv2.destroyAllWindows()
write_img_array_to_video(img_arr, fps, 'VORTEX.avi')
# cap.release()

