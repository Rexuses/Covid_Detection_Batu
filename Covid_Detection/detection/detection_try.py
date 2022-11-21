import time

import cv2
import numpy as np


# 1) write a function that finds first nonzero pixel row and column.
# 2) write a function that crop the image given row to row and column to column.


def crop_image(img, row1, row2, col1, col2):
    crop_img = img[row1:row2, col1:col2]
    return crop_img


def up_down_nonzero_pixel(img):
    find = False
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] != 0:
                find = True
                break
        if find:
            break

    return row


def down_up_nonzero_pixel(img):
    find = False
    for row in range(img.shape[0] - 1, 0, -1):
        for col in range(img.shape[1] - 1, 0, -1):
            if img[row, col] != 0:
                find = True
                break
        if find:
            break

    return row


def left_right_nonzero_pixel(img):
    find = False
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if img[row, col] != 0:
                find = True
                break
        if find:
            break

    return col


def right_left_nonzero_pixel(img):
    find = False
    for col in range(img.shape[1] - 1, 0, -1):
        for row in range(img.shape[0] - 1, 0, -1):
            if img[row, col] != 0:
                find = True
                break
        if find:
            break
    return col



lung_mask = cv2.imread('lung_mask.png', cv2.IMREAD_GRAYSCALE)
org = cv2.imread('org.png', cv2.IMREAD_GRAYSCALE)
ggo_mask = cv2.imread('ggo_mask.png', cv2.IMREAD_GRAYSCALE)


result_lung = cv2.bitwise_and(org, org, mask=lung_mask)
cropped = crop_image(result_lung, up_down_nonzero_pixel(result_lung), down_up_nonzero_pixel(result_lung),
                     left_right_nonzero_pixel(result_lung), right_left_nonzero_pixel(result_lung))
cv2.imwrite("deneme.png",cropped)
time.sleep(2)
crop_images = cv2.imread('deneme.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('org', org)
cv2.imshow('cropped', cropped)

bos_list = np.zeros((crop_images.shape[0], crop_images.shape[1]), dtype='uint8')
row_deneme = 10
for i in range(crop_images.shape[0]-row_deneme+1):
    for j in range(crop_images.shape[1]-row_deneme+1):
        if np.mean(crop_images[i:i+row_deneme,j:j+row_deneme]) > 145:
            bos_list[i][j] = 0
        else:
            bos_list[i][j] = 1

result_lung3 = cv2.bitwise_and(crop_images,crop_images,mask=bos_list)
cv2.imshow("result_lung3", result_lung3)
cv2.waitKey(0)
