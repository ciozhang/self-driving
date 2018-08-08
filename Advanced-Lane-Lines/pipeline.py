import cv2
import numpy as np
from calibration import undistort
from skimage import morphology


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def select_line_color(image):
    cvt_image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) # h l s
    # yellow mask
    yellow_lower = np.uint8([19,80,5]) # h l s
    yellow_upper = np.uint8([40,200,235]) 
    yellow_mask = cv2.inRange(cvt_image,yellow_lower,yellow_upper)
    # white mask
    white_lower = np.array([5,190,50])
    white_upper = np.array([180,255,255])
    white_mask = cv2.inRange(cvt_image,white_lower,white_upper)
    # combine mask
    mask = cv2.bitwise_or(yellow_mask,white_mask)
    return cv2.bitwise_and(image,image,mask=mask)


def ROI(img):
    h,w = img.shape[:2]
    mask = np.zeros_like(img)
    vertices = np.array([[(0,h*13/14),(0,h*5/8),(w,h*5/8),(w,h*13/14)]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,1)

    masked_image = cv2.bitwise_and(img,mask)
    return masked_image


def clean(img,min_sz):
    cleaned = morphology.remove_small_objects(img.astype('bool'),min_size=min_sz,connectivity=2)
    return cleaned

