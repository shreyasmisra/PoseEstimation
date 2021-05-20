import cv2 as cv
import mediapipe as mp
import PoseEstimatorClass as p

#video_path = <insert path>
#img_path = <insert path>

image_mode = False
#detection_confidence = 0.8
#tracking_confidence = 0.5
detector_model = p.PoseDetector(static_image_mode=image_mode)

if image_mode:
    img = cv.imread(img_path)
    img = detector_model.find_pose(img)
    cv.imshow("Image",img)
    
if not image_mode:
    #cap = cv.VideoCapture(video_path)
    cap = cv.VideoCapture(0) # For webcam
    while True:
        retval,image = cap.read()
        image = detector_model.find_pose(image)
        landmarks = detector_model.get_landmarks(image,to_print=False)
        image = cv.flip(image,1)
        cv.imshow("Image", image)
        k = cv.waitKey(1)
    
        if k==32: #space bar
            print("Program Terminated")
            break