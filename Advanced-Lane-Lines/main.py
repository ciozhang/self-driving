import os
import cv2
from moviepy.editor import VideoFileClip
from pipeline import process_image
from calibration import calibrate




#challenge_output = 'challenge_out.mp4'
#clip = VideoFileClip('challenge_video.mp4').subclip(0,5)
#challenge_clip = clip.fl_image(process_image)
#challenge_clip.write_videofile(challenge_output, audio=False)

cal_images_dir = './camera_cal/calibration*.jpg'
ret, mtx, dist, rvecs, tvecs=calibrate(cal_images_dir)

for img_name in os.listdir("test_images/"):
    print(img_name)
    img=cv2.imread("test_images/"+img_name)
    out=process_image(img, mtx, dist)
    cv2.imwrite(img_name[:-4]+"_out.jpg", out)
