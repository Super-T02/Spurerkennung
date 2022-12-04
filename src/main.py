import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import time

import calib as cal
import perspective_transform as per
import sliding_window as slw

class Main():
    # Window size
    WIN_X = 1280
    WIN_Y = 720

    # Gauss settings
    GAUSS_KERNEL = (5, 5)

    # Canny settings
    CANNY_LOWER = 100
    CANNY_UPPER = 150

    # Hough Settings
    HOUG_RHO = 6
    HOUG_THETA = 90
    HOUGH_THRESHOLD = 100
    HOUGH_MIN_LINE_LENGTH = 3
    HOUGH_MAX_LINE_GAP = 2

    # FIX Point Hough
    LEFT_FIX = (200, 720)
    RIGHT_FIX = (1200, 720)


    def __init__(self, path, roi, canny_lower = None, canny_upper = None, thresh = None, debug = False):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')
        if path is None or roi is None: return print('No path or roi given')
        self.calib = cal.Calibration(False)
        self.transformation = per.Transformation(False)
        self.sliding_win = slw.SlidingWindow(thresh)
        
        self.path = path
        self.roi = roi
        self.debug = debug

        # TODO: Refactor Hough
        self.left_line = {
            'FIT_X': [],
            'PLOT_Y': [],
        }
        self.right_line = {
            'FIT_X': [],
            'PLOT_Y': [],
        }

        self.canny_lower = canny_lower if canny_lower is not None else self.CANNY_LOWER
        self.canny_upper = canny_upper if canny_upper is not None else self.CANNY_UPPER
        
    def startVideo(self, mode=0, hough=False, show_areal=False):
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
            self.equalized_img = frame
            if mode == 1:
                frame = self.sliding_win.execute(frame)
            else:
                frame = self._preprocess_default(frame, hough=hough) if not show_areal else self._preprocess_areal_view(frame)

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

    def _preprocess_areal_view(self, img):
        img, _ = self.transformation.transform_image_perspective(img)
        img = self._gauss(img)
        img = self._canny(img)
        return img

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
    main1 = Main(videoHarder, roi_videoHarder, canny_lower=15, canny_upper=100, debug=True)
    main2 = Main(videoHardest, roi_videoHardest)

    # Mode:
    # - 0: Hough
    # - 1: Sliding window

    # main.startVideo()
    main.startVideo(mode=1)
    # main.startVideo(hough=True, show_areal=True)
    # main.startVideo(hough=True)
    # main1.startVideo()
    # main1.startVideo(hough=True)
    # main2.startVideo()
    # main2.startVideo(hough=True)
    