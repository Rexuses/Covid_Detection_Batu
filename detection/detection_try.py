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



lung_mask = cv2.imread("lung_mask.png", cv2.IMREAD_GRAYSCALE)
org = cv2.imread("org.png", cv2.IMREAD_GRAYSCALE)
ggo_mask = cv2.imread("ggo_mask.png", cv2.IMREAD_GRAYSCALE)


result_lung = cv2.bitwise_and(org, org, mask=lung_mask)

cropped = crop_image(result_lung, up_down_nonzero_pixel(result_lung), down_up_nonzero_pixel(result_lung),
                     left_right_nonzero_pixel(result_lung), right_left_nonzero_pixel(result_lung))


cv2.imwrite("deneme.png", cropped)
time.sleep(2)
crop_images = cv2.imread('deneme.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('org', org)
cv2.imshow('cropped', cropped)

mean= 0
count= 0

print(crop_images.shape[1])

for i in range(crop_images.shape[0]):
    for j in range(crop_images.shape[1]):
        if i != 0:
            count+=1
            mean+=cropped[i][j]
mean = mean/count
print(mean)

gamma = 2
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv2.LUT(crop_images, lookUpTable)
img_gamma_corrected = cv2.hconcat([cropped, res])
cv2.imshow("Gamma correction", img_gamma_corrected)

blur = cv2.GaussianBlur(res,(5,5),0)
cv2.imshow("blur",blur)

ret3,th3 = cv2.threshold(res,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret3)
cv2.imshow("otsu",th3)
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
