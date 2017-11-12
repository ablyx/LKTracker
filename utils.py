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

def write_img_array_to_video(img_arr, fps, video_path):
    num_frames = len(img_arr)
    print(num_frames)
    height,width,layers = img_arr[0].shape
    print("Height", height)
    print("Width", width)
    print("Layers", layers)
    video = cv2.VideoWriter(video_path,cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V'),fps,(width,height))

    for i in range(num_frames):
        video.write(img_arr[i])
    video.release()
    cv2.destroyAllWindows()
    return video
