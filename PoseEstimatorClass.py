import cv2 as cv
import mediapipe as mp

class PoseDetector:
    def __init__(self,static_image_mode=False,smooth_landmarks=True,detection_confidence=0.7,track_confidence=0.5):
        """        
        Parameters
        ----------
        static_image_mode : bool, optional
            static_image_mode tells the pose class whether an image part of a video is passed or not
            True when a single image is passed into the pose class. The default is False.
        smooth_landmarks : bool, optional
            if set to true, the solution filters pose landmarks across different input images to reduce jitter, 
            but ignored if static_image_mode is also set to true. The default is True.
        detection_confidence : bool,optional
            Minimum confidence value ([0.0, 1.0]) from the person-detection model for the detection to be considered successful. The default is 0.7.
        track_confidence : TYPE, optional
            Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the pose landmarks to be considered tracked successfully, 
            or otherwise person detection will be invoked automatically on the next input image. . The default is 0.5.

        Returns
        -------
        None.
        """
        
        self.mode = static_image_mode
        self.smooth = smooth_landmarks
        self.detect_con = detection_confidence
        self.track_con = track_confidence
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode = self.mode,
                                      smooth_landmarks = self.smooth,
                                      min_detection_confidence = self.detect_con,
                                      min_tracking_confidence = self.track_con)
    
    def find_pose(self,img,draw=True,save_img_path=None):
        """
        Parameters
        ----------
        img : image variable
            Image on which the pose has to be estimated.
        draw : bool, optional
            Whether to draw the pose on the image/video. The default is True.
        save_img_path : bool, optional
            Whether to save the image. Works only when self.mode = True. 
            To save the image assign the path to the save_img_path variable. The default is None.

        Returns
        -------
        img : list
            The input image
        """
        
        if self.mode:
            save_img_path = None
            
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        
        if self.results.pose_landmarks:
            if draw: 
                if not self.mode:
                    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
                    
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            if save_img_path:
                cv.write(save_img_path+".png",img)
        
        else:
            print("Pose detection failed. No objects/images/poses found")
        
        return img
        
    def get_landmarks(self,img,draw=True,to_print=False):
        """
        Parameters
        ----------
        img : image 
            The input image/snapshot of a video.
        draw : bool, optional
            Whether to draw the pose on the image/video. The default is True.
        to_print : bool, optional
            True when the landmark parameters -- x,y,z and visibility -- have to be printed. The default is False.

        Returns
        -------
        landmarks : list
            List of ID and centers of each landmarks in the frame.
        """
        
        landmarks= []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                
                if to_print:   
                    print("Landmark ID: ",idx,"\nLocation:\n ", lm,"\n")
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([idx, cx, cy])
                
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return landmarks
    