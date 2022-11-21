import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import time

import calib as cal

class Main():
    # Window size
    WIN_X = 1280
    WIN_Y = 720

    def __init__(self, path):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')
        self.calib = cal.Calibration(False)
        self.path = path
        
    def startVideo(self):
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
            frame = self._preprocess(frame)

            # Do operations on the frame
            gray = frame
            gray = cv.resize(gray, (self.WIN_X, self.WIN_Y))
            font = cv.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()

            # Calculate Frame Rate
            fps, prev_frame_time = self._calcFPS(prev_frame_time, new_frame_time)

            # Put fps on the screen
            cv.putText(gray, fps, (7, 21), font, 1, (100, 100, 100), 2, cv.LINE_AA)

            cv.imshow('Video', gray)

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

    def _preprocess(self, img):
        # Equalize the image
        img = self.calib.equalize(img)

        # Convert to grayscale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # # Apply Gaussian blur
        # blur = cv.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv.Canny(img, 100, 150)

        return edges

if __name__ == '__main__':
    # Path to video
    video = "img/Udacity/project_video.mp4"
    videoHarder = "img/Udacity/challenge_video.mp4"
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    
    # Start the program
    main = Main(video)
    main.startVideo()
    