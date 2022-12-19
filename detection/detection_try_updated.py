import time

import cv2
import numpy as np
import os

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



lung_mask = cv2.imread("lung_mask6.png", cv2.IMREAD_GRAYSCALE)
org = cv2.imread("org6.png", cv2.IMREAD_GRAYSCALE)


result_lung = cv2.bitwise_and(org, org, mask=lung_mask)

cropped = crop_image(result_lung, up_down_nonzero_pixel(result_lung), down_up_nonzero_pixel(result_lung),
                     left_right_nonzero_pixel(result_lung), right_left_nonzero_pixel(result_lung))

cv2.imshow("org",org)
cv2.imshow("result_lung",result_lung)

cv2.imwrite("deneme.png", cropped)
time.sleep(2)
crop_images = cv2.imread('deneme.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('cropped', cropped)


mean= 0
count= 0

print(crop_images.shape[1])



gamma = 0.85
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv2.LUT(crop_images, lookUpTable)
img_gamma_corrected = cv2.hconcat([cropped, res])
cv2.imshow("Gamma correction", img_gamma_corrected)
cv2.imshow("gamma",res)
diff=cv2.subtract(res,cropped)
cv2.imshow("diff",diff)


blur = cv2.GaussianBlur(diff,(5,5),0)
cv2.imshow("blur",blur)
cv2.imwrite("diff.png",diff)


cv2.waitKey(0)
