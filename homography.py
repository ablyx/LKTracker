import cv2
import copy
import numpy as np

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

# board pts is in the form x,y
def new_warp_subimage(image, subimage, board_pts):
    # warping vortex to board
    # orig_pts is of form [(u0,v0), p1, ...]
    # pts is of form [(u1, v1, u1t, v1t), (u2, v2, u2t, v2t), ...]

    pts = []
    rows, cols, s = subimage.shape
    vortex_pts = [(0,0), (cols-1, 0), (0, rows-1), (cols-1, rows-1)] # in the form of x,y
    for i, vp in enumerate(vortex_pts):
        u, v = vp
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
    for r, row in enumerate(subimage): # y
        for c, pixel in enumerate(row): # x
            # for vortex, make corners transparent
            if r < rows/4 or r > 3*rows/4 or c < cols/4 or c > 3*cols/4:
                dist = sum(list(pixel))
                if dist < DIST_THRESHOLD:
                    continue
            # if list(pixel) == [0,0,0] or list(pixel) == [255,255,255]:
            #     continue
            # warp x,y
            new_coord = warp((c,r),h)
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
                # print('failed')
                pass
    # i want to get rectangle from board pts to blur
    # do median blur on 4 corners to remove purple dot first
    # do it repeatedly because dot is too big
    xcoords = map(lambda p:p[0], board_pts)
    ycoords = map(lambda p:p[1], board_pts)
    p1 = (min(xcoords), min(ycoords))
    p2 = (max(xcoords), min(ycoords))
    p3 = (min(xcoords), max(ycoords))
    p4 = (max(xcoords), max(ycoords))
    corners = [p1,p2,p3,p4]

    # tweak this numbers maybe.
    CORNER_BLUR_WINDOW = 15
    CONV_WINDOW = 33
    WW = CONV_WINDOW/2
    MEDIAN_ITER = 3
    dots = copy.deepcopy(image)
    for _ in range(MEDIAN_ITER):
        for corner in board_pts:
            x,y = corner
            x = int(x)
            y = int(y)
            for i in range(y-CORNER_BLUR_WINDOW/2, y+CORNER_BLUR_WINDOW/2): # y
                for j in range(x-CORNER_BLUR_WINDOW/2, x+CORNER_BLUR_WINDOW/2): # x
                    # in case out of bounds
                    try:             
                        window = dots[i-WW:i+WW+1, j-WW:j+WW+1]
                        # print(np.average(window))
                        image[i,j] = int(np.average(window))
                    except:
                        # print('error')
                        continue
        dots = copy.deepcopy(image)

    print('dots removed')

    # blur vortex, dont blur corners
    box = copy.deepcopy(image)
    AVERAGE_BLUR_WINDOW = 5
    WW = AVERAGE_BLUR_WINDOW/2
    for r, row in enumerate(subimage): # y
        for c, pixel in enumerate(row): # x
            # for vortex, make corners transparent
            if r < rows/4 or r > 3*rows/4 or c < cols/4 or c > 3*cols/4:
                dist = sum(list(pixel))
                if dist < DIST_THRESHOLD:
                    continue
                else:
                    new_coord = warp((c,r),h)
                    x, y, z = new_coord
                    x = int(round(x))
                    y = int(round(y))
                    try:
                        # rounding error might cause some pixels not to show
                        for i in range(-2,3):
                            for j in range(-2,3):
                                ny = y+i
                                nx = x+j
                                avg = np.average(box[ny-WW:ny+WW+1, nx-WW:nx+WW+1])
                                image[(ny,nx)] = int(avg)
                    except:
                        pass

            new_coord = warp((c,r),h)
            x, y, z = new_coord
            x = int(round(x))
            y = int(round(y))
            try:
                # rounding error might cause some pixels not to show
                for i in range(2):
                    for j in range(2):
                        ny = y+i
                        nx = x+j
                        avg = np.average(box[ny-WW:ny+WW+1, nx-WW:nx+WW+1])
                        image[(ny,nx)] = int(avg)
            except:
                # print('failed')
                continue
    print('blurred')

    return image


def warp(old_coord, h):
    x, y = old_coord
    new_coord = np.dot(h,[[x,], [y,], [1,]])
    norm_new_coord = (1 / new_coord[-1]) * new_coord
    return norm_new_coord