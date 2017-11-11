## Defines utility functions

import numpy as np
import cv2

# Utility functions
def display_img(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    return

def read_img(path):
    img = cv2.imread(path, 0)
    return img
