from homography import new_warp_subimage
from utils import *
import numpy as np
import cv2


cap = cv2.VideoCapture('vortex.mp4')
fps = cap.get(5)
# params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (13,13),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# print('p0', p0)
# rectangle
# in the form x, y
orig_pts = [(286,129),(282,329),(535,135),(529,329)]
p0 = np.array(map(lambda p: np.array(p), orig_pts))
p0 = p0.astype('float32')

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
img_arr = [firstFrameVortex,]
# cv2.imshow('vortex',firstFrameVortex)

counter = 1

while(counter<270):
    ret,frame = cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    board_pts = list(p1)
    v = counter % 12
    vortex = vortexes[v]
    frameVortex = new_warp_subimage(frame, vortex, board_pts)
    img_arr.append(frameVortex)
    # cv2.imshow('vortex',frameVortex)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = p1
    counter = counter + 1
    print(counter)

# cv2.destroyAllWindows()
write_img_array_to_video(img_arr, fps, 'VORTEX_FINAL.avi')
# cap.release()

