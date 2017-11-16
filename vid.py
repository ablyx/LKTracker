from homography import warp_subimage

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
