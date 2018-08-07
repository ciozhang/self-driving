import cv2
import numpy as np
from calibration import undistort


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def select_line_color(image):
    cvt_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) # h l s
    # yellow mask
    yellow_lower = np.uint8([10,0,100]) # h l s
    yellow_upper = np.uint8([40,255,255]) 
    yellow_mask = cv2.inRange(cvt_image,yellow_lower,yellow_upper)
    # white mask
    white_lower = np.array([0,200,0])
    white_upper = np.array([255,255,255])
    white_mask = cv2.inRange(cvt_image,white_lower,white_upper)
    # combine mask
    mask = cv2.bitwise_or(yellow_mask,white_mask)
    return cv2.bitwise_and(image,image,mask=mask)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def linear_fit(x, a, b):
    return a * x + b


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    right_x,right_y,left_x,letf_y=[],[],[],[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



def process_image(image, mtx, dist):
    dst=undistort(image,mtx,dist)
    img_color_select = select_line_color(image)
    img_gray = grayscale(img_color_select)
    img_blur = gaussian_blur(img_gray, 15)
    img_edge = canny(img_blur, low_threshold=50, high_threshold=150)
    rows, cols = image.shape[:2]
    vertices = np.array([[[cols*0.1, rows*0.95],[cols*0.4, rows*0.6],[cols*0.9, rows*0.95],[cols*0.6, rows*0.6]]],dtype=np.int32)
    img_masked = region_of_interest(img_edge, vertices)
    img_lined = hough_lines(img_masked, rho=2, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=50)
    img_weighted = weighted_img(img_lined, image, α=0.8, β=1., λ=0.)
    return img_weighted

