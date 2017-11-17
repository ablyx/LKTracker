import cv2
import copy
import numpy as np
import scipy.signal
# u is x v is y
img = cv2.imread('test/bottle1.jpg')
to_warp = cv2.imread('test/nobottle.jpg')
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("to_warp", cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.imshow('to_warp', to_warp)

# bottle points
# bottle1 to nobottle
u1 = 717
v1 = 478

u2 = 785
v2 = 478

u3 = 733
v3 = 502

u4 = 803
v4 = 502

u1t = 656
v1t = 485

u2t = 707
v2t = 467

u3t = 702
v3t = 514

u4t = 753
v4t = 493

# mouse points
# mouse1 to mouse 2
# u1 = 318
# v1 = 767
#
# u2 = 320
# v2 = 883
#
# u3 = 155
# v3 = 814
#
# u4 = 171
# v4 = 930
#
# u1t = 235
# v1t = 384
#
# u2t = 247
# v2t = 492
#
# u3t = 69
# v3t = 353
#
# u4t = 64
# v4t = 461

# # mouse points
# # mouse2 to mouse3
# u1 = 631
# v1 = 455
#
# u2 = 737
# v2 = 468
#
# u3 = 666
# v3 = 548
#
# u4 = 778
# v4 = 559
#
# u1t = 637
# v1t = 411
#
# u2t = 626
# v2t = 357
#
# u3t = 733
# v3t = 409
#
# u4t = 717
# v4t = 354
pts = [(u1, v1, u1t, v1t), (u2, v2, u2t, v2t), (u3, v3, u3t, v3t), (u4, v4, u4t, v4t)]

def small_mat(u, v, ut, vt):
    r1 = np.array([u, v, 1, 0, 0, 0, -ut*u, -ut*v, -ut])
    r2 = np.array([0, 0, 0, u, v, 1, -vt*u, -vt*v, -vt])
    return np.vstack((r1, r2))


def big_mat(m1, m2, m3, m4):
    return np.vstack((m1, m2, m3, m4))


def homo_mat(mat):
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    V = np.transpose(V)
    h = (1/V[-1,-1])*V[:,-1]
    return np.reshape(h, (3,3))
"""
# orig pts are saved from the first frame
# to_warp_pts are the tracked points
def warp_subimage(image, subimage, orig_pts, to_warp_pts):
    #calculate offet from to left corner of subimage
    print('op0', orig_pts[0])
    # offset = orig_pts[0] # this is in the form x, y
    # print('offset',offset)
    # orig_pts is of form [(u0,v0), p1, ...]
    # pts is of form [(u1, v1, u1t, v1t), (u2, v2, u2t, v2t), ...]
    pts = []
    # print('to_warp_pts', to_warp_pts)
    # print('to_warp_pts', list(map(lambda p:list(p), to_warp_pts)))
    
    for i, op in enumerate(orig_pts):
        u, v = op
        # print('i', to_warp_pts[i])
        ut, vt = to_warp_pts[i]
        pts.append((u, v, ut, vt))

    u, v, ut, vt = pts[0]
    # print(u, v, ut, vt)
    bm = small_mat(u, v, ut, vt)
    for pt in pts[1:]:
        u, v, ut, vt = pt
        sm = small_mat(u, v, ut, vt)
        bm = np.vstack((bm,sm))
    h = homo_mat(bm)

    rows, cols, s = subimage.shape
    # test if h works
    op = orig_pts
    for i, pt in enumerate(pts):
        u, v, ut, vt = pt
        r, c = op[i]
        print('ut vt:', ut, vt)
        print('ut vt:', warp((u, v), h))
        print('r c:' , (r,c))
        print('hr hc:', warp((r,c), h))

    # print(h)
    
    for r, row in enumerate(subimage):
        for c, pixel in enumerate(row):
            new_coord = warp((c,r),h)
            x, y, z = new_coord
            x = int(round(x))
            y = int(round(y))
            if (r,c) == (0,0):
                top_left_x, top_left_y = to_warp_pts[0]
                offset = [top_left_x-x, top_left_y - y]
                print('offset', offset)
                # print('topleftwarp',(y+int(offset[0]), x+int(offset[1])))
                # print(to_warp_pts[0])
            # image_warped_coord = (x+int(offset[1]), y+int(offset[0]))
            image_warped_coord = (y+int(offset[1]), x+int(offset[0]))
            if (r,c) == (0,0):
                print('image_warped_coord 0 0', image_warped_coord, (y,x))
            if (r,c) == (rows-1,0):
                print('image_warped_coord rows 0', image_warped_coord, (y,x))
            if (r,c) == (0,cols-1):
                print('image_warped_coord 0 cols', image_warped_coord, (y,x))
            if (r,c) == (rows-1,cols-1):
                print('image_warped_coord rows cols', image_warped_coord, (y,x))
            
            try:
                image[image_warped_coord] = pixel
            except:
                # print('fail')
                pass
    return image
"""
def new_warp_subimage(image, subimage, board_pts):
    # warping vortex to board
    #calculate offet from to left corner of subimage
    # print('op0', orig_pts[0])
    # offset = orig_pts[0] # this is in the form x, y
    # print('offset',offset)
    # orig_pts is of form [(u0,v0), p1, ...]
    # pts is of form [(u1, v1, u1t, v1t), (u2, v2, u2t, v2t), ...]
    pts = []
    # print('to_warp_pts', to_warp_pts)
    # print('to_warp_pts', list(map(lambda p:list(p), to_warp_pts)))
    rows, cols, s = subimage.shape
    vortex_pts = [(0,0), (rows-1, 0), (0, cols-1), (rows-1, cols-1)] # in the form of x,y 
    for i, vp in enumerate(vortex_pts):
        u, v = vp
        # print('i', to_warp_pts[i])
        ut, vt = board_pts[i]
        pts.append((u, v, ut, vt))

    u, v, ut, vt = pts[0]
    # print(u, v, ut, vt)
    bm = small_mat(u, v, ut, vt)
    for pt in pts[1:]:
        u, v, ut, vt = pt
        sm = small_mat(u, v, ut, vt)
        bm = np.vstack((bm,sm))
    h = homo_mat(bm)

    DIST_THRESHOLD = 300
    # print(h)
    for r, row in enumerate(subimage):
        for c, pixel in enumerate(row):
            # for vortex, make corners transparent
            if r < rows/4 or r > 3*rows/4 or c < cols/4 or c > 3*cols/4:
                dist = sum(list(pixel))
                if dist < DIST_THRESHOLD:
                    continue
            # if list(pixel) == [0,0,0] or list(pixel) == [255,255,255]:
            #     continue
            new_coord = warp((r,c),h)
            x, y, z = new_coord
            x = int(round(x))
            y = int(round(y))
            image_warped_coord = (y,x)
            try:
                image[image_warped_coord] = pixel
                # rounding error might cause some pixels not to show
                for i in range(2):
                    for j in range(2):
                        # print('test', y, x)
                        image[(y+i,x+j)] = pixel
            except:
                print('failed')
    # i want to get rectangle from board pts to blur
    # do median blur on 4 corners to remove purple dot first
    # do it repeatedly because dot is too big
    # then blur the rest
    # i think only blur the sides? centre no need blur
    xcoords = map(lambda p:p[0], board_pts)
    ycoords = map(lambda p:p[1], board_pts)
    p1 = (min(xcoords), min(ycoords))
    p2 = (min(xcoords), max(ycoords))
    p3 = (max(xcoords), min(ycoords))
    p4 = (max(xcoords), max(ycoords))
    corners = [p1,p2,p3,p4]
    CORNER_BLUR_WINDOW = 33
    CONV_WINDOW = 15
    WW = CONV_WINDOW/2
    MEDIAN_ITER = 3
    dots = copy.deepcopy(image)
    for _ in range(MEDIAN_ITER):
        for corner in corners:
            x,y = corner
            x = int(x)
            y = int(y)
            for i in range(y-CORNER_BLUR_WINDOW/2, y+CORNER_BLUR_WINDOW/2):
                for j in range(x-CORNER_BLUR_WINDOW/2, x+CORNER_BLUR_WINDOW/2):
                    # in case out of bounds
                    try:             
                        window = dots[i-WW:i+WW+1, j-WW:j+WW+1]
                        # print(window)
                        image[i,j] = np.median(window)
                    except:
                        print('error')
                        continue
        dots = copy.deepcopy(image)
    return image


def warp_img(img, pts):
    u, v, ut, vt = pts[0]
    bm = small_mat(u, v, ut, vt)
    for pt in pts[1:]:
        u, v, ut, vt = pt
        sm = small_mat(u, v, ut, vt)
        bm = np.vstack((bm,sm))
    h = homo_mat(bm)
    r, c, s = img.shape
    buffer = 2000
    himg = np.zeros((r+2*buffer, c+2*buffer, 3))
    print(himg.shape)
    print(img.shape)
    for r, row in enumerate(img):
        for c, pixel in enumerate(row):
            new_coord = warp((c,r),h)
            x, y, z = new_coord
            x = int(x)
            y = int(y)
            # print(x,y)
            try:
                himg[y+buffer,x+buffer] = pixel
            except:
                pass
    return himg


def warp(old_coord, h):
    x, y = old_coord
    new_coord = np.dot(h,[[x,], [y,], [1,]])
    norm_new_coord = (1 / new_coord[-1]) * new_coord
    return norm_new_coord


def test(pts, img, to_warp):
    u, v, ut, vt = pts[0]
    bm = small_mat(u, v, ut, vt)
    for pt in pts[1:]:
        u, v, ut, vt = pt
        sm = small_mat(u, v, ut, vt)
        bm = np.vstack((bm,sm))
    h = homo_mat(bm)
    for pt in pts:
        u, v, ut, vt = pt
        new_coord = warp((u,v), h)
        x, y, z = new_coord
        x = int(x)
        y = int(y)
        print(x,y)
        print(ut,vt)
        # cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        # cv2.circle(to_warp, (x, y), 1, (0, 0, 255), -1)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("to_warp", cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.imshow('to_warp', to_warp)

def cvhomo(img, pts):
    src = np.array(map(lambda x: x[:2], pts))
    dst = np.array(map(lambda x: x[2:], pts))
    h, status = cv2.findHomography(src, dst)
    warped_img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
    return warped_img


if __name__ == "main":
    test(pts, img, to_warp)
    # warped_img = warp_img(img, pts)
    warped_img = cvhomo(img, pts)
    print('done')
    cv2.namedWindow("warped_img", cv2.WINDOW_NORMAL)
    cv2.imshow('warped_img', warped_img)
    cv2.imwrite('cvwarped_img_bottletonobottle.jpg', warped_img)
    cv2.waitKey(0)