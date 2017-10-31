import cv2
import numpy as np


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
        cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        cv2.circle(to_warp, (x, y), 1, (0, 0, 255), -1)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("to_warp", cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.imshow('to_warp', to_warp)

test(pts, img, to_warp)
warped_img = warp_img(img, pts)
print('done')
cv2.namedWindow("warped_img", cv2.WINDOW_NORMAL)
cv2.imshow('warped_img', warped_img)
cv2.imwrite('warped_img_bottletonobottlebig2000.jpg', warped_img)
cv2.waitKey(0)