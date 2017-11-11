import numpy as np
import cv2
from utils import display_img, read_img

## Takes an image and downsamples by a factor of 2
def down_sample(im):
    num_rows = im.shape[0]
    num_cols = im.shape[1]
    new_img = np.zeros((num_rows/2, num_cols/2), dtype=np.float32)
    # Iterate through the image in blocks of 4 pixels
    new_row_idx = 0
    new_col_idx = 0

    for row_idx in range(0, num_rows, 2):
        for col_idx in range(0, num_cols, 2):
            downsampled_pix_val = int(round((int(im[row_idx, col_idx]) + int(im[row_idx+1, col_idx]) + int(im[row_idx, col_idx+1]) + int(im[row_idx+1, col_idx+1])) / 4))
            new_img[new_row_idx, new_col_idx] = downsampled_pix_val
            new_col_idx += 1
        # Reset new_col_idx
        new_col_idx = 0
        # Update new_row_idx
        new_row_idx += 1

    return new_img

# # Utility functions
# def display_img(img, title):
#     cv2.imshow(title, img)
#     cv2.waitKey(0)
#     return
#
# def read_img(path):
#     img = cv2.imread(path, 0)
#     return img

if __name__ == '__main__':
    # Read image
    img = read_img('test.jpg')
    # display_img(img, 'Test')
    # display_img(img, 'Test Image')

    # Down sample
    new_img = down_sample(img)
    print(new_img.shape)
    display_img(new_img, 'DownSampled')
    cv2.imwrite('new_img.jpg', new_img)
    print(new_img)
    # Output downsampled image
