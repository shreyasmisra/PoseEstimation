import cv2 as cv
import mediapipe as mp
import PoseEstimatorClass as p

# video_path = " "
# img_path = " "

cap = cv.VideoCapture(0)
detector_model = p.PoseDetector()

while True:
    retval,image = cap.read()
    image = detector_model.find_pose(image,)
    
    image = cv.flip(image,1)
    cv.imshow("Image", image)
    k = cv.waitKey(1)
    
    if k==32: #space bar
        print("Program Terminated")
        break