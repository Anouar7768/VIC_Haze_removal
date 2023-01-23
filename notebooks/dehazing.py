
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Write a function to dehaze an image
def dehaze_copilot(img):
    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract the V channel
    img_v = img_hsv[:, :, 2]
    # Create a kernel for the box filter
    kernel = np.ones((15, 15), np.float32) / 225
    # Apply the box filter to the V channel
    img_v_blur = cv2.filter2D(img_v, -1, kernel)
    # Calculate the transmission map
    img_t = 1 - 0.95 * (img_v / img_v_blur)
    # Calculate the atmospheric light
    img_atm = np.mean(img_v)
    # Calculate the radiance map
    img_r = (img - img_atm) / img_t + img_atm
    # Clip the values in the radiance map
    img_r[img_r > 255] = 255
    img_r[img_r < 0] = 0
    # Convert the radiance map back to BGR
    img_dehazed = cv2.cvtColor(img_r.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img_dehazed

