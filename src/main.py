import cv2 as cv
import os
import time

import calib as cal
import perspective_transform as per
import sliding_window as slw
import hough_transformation as hou

class Main():
    # Window size
    WIN_X = 1280
    WIN_Y = 720

    def __init__(self, path, roi, canny_lower = None, canny_upper = None, thresh = None, debug = False):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')
        if path is None or roi is None: return print('No path or roi given')
        
        # Define the objects
        self.calib = cal.Calibration(debug=debug)
        self.transformation = per.Transformation(debug=debug)
        self.sliding_win = slw.SlidingWindow()
        self.hough = hou.HoughTransformation(roi, debug = debug)
        
        # Define the variables
        self.path = path
        
    def startVideo(self, mode=0):
        if not os.path.exists(self.path):
            return print('Video not found')

        # Load video
        video = cv.VideoCapture(self.path)
        prev_frame_time = 0
        new_frame_time = 0

        # While the video is running
        while(video.isOpened()):
            ret, frame = video.read()

            # Break if video is finish or no input
            if not ret:
                break

            # Do here the image processing
            frame = cv.resize(frame, (self.WIN_X, self.WIN_Y))
            
            # Equalize the image
            frame = self.calib.equalize(frame)
            
            # Choose the mode
            if mode == 0:
                frame = self.hough.execute(frame)
            elif mode == 1:
                frame = self.sliding_win.execute(frame)
            else:
                return print('Mode not found')

            # Do operations on the frame
            if frame is not None:
                #transformed = self.transformation.transform_image_perspective(frame)
                font = cv.FONT_HERSHEY_SIMPLEX
                new_frame_time = time.time()

                # Calculate Frame Rate
                fps, prev_frame_time = self._calcFPS(prev_frame_time, new_frame_time)

                # Put fps on the screen
                cv.putText(frame, fps, (7, 21), font, 1, (100, 100, 100), 2, cv.LINE_AA)

                cv.imshow('Video', frame)

            # press 'Q' for exit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


        # Stop video and close window
        video.release()
        cv.destroyAllWindows()

    def _calcFPS(self, prev_frame_time, new_frame_time):
        # Calculate Frame Rate
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        return fps, prev_frame_time


if __name__ == '__main__':
    # Path to video
    video = "img/Udacity/project_video.mp4"
    roi_video = [
            (300, - 75), # BOTTOM LEFT
            (-55, 72), # TOP LEFT
            (78, 72), # TOP RIGHT
            (-150, - 75), # BOTTOM RIGHT
    ]
    videoHarder = "img/Udacity/challenge_video.mp4"
    roi_videoHarder = [
            (300, - 75),
            (-40, 90),
            (50, 80),
            (-325, - 75),
    ]
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    roi_videoHardest = [
            (300, - 75),
            (-60, 75),
            (40, 75),
            (150, - 75),
    ]
    
    # Start the program
    main = Main(video, roi_video, 1, debug=False) # canny_lower=50, canny_upper=100 if you change the order of areal view preprocessing 
    # main1 = Main(videoHarder, roi_videoHarder, canny_lower=15, canny_upper=100, debug=True)
    # main2 = Main(videoHardest, roi_videoHardest)

    # Mode:
    # - 0: Hough
    # - 1: Sliding window

    # main.startVideo()
    main.startVideo(mode=0)
    # main.startVideo(hough=True, show_areal=True)
    # main.startVideo(hough=True)
    # main1.startVideo()
    # main1.startVideo(hough=True)
    # main2.startVideo()
    # main2.startVideo(hough=True)
    