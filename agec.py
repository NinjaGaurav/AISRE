import numpy as np
import cv2
import argparse

def agcwd(image, w=0.5):
    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image


def extract_value_channel(color_image):
    color_image = color_image.astype(np.float32) / 255.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return np.uint8(v * 255)


def get_pdf(gray_image):
    height, width = gray_image.shape
    pixel_count = height * width
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist / pixel_count


def set_value_channel(color_image, value_channel):
    value_channel = value_channel.astype(np.float32) / 255
    color_image = color_image.astype(np.float32) / 255.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image[:, :, 2] = value_channel
    color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
    return color_image

def agec_init(img):

    enhanced_image = agcwd(img) 
    gamma = 0.75
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(enhanced_image, lookUpTable)
    return res
