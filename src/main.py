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
        
    def startVideo(self, hugs=False):
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
            frame = self._preprocess(frame, hugs=hugs)

            # Do operations on the frame
            gray = frame
            if gray is not None:
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

    def _preprocess(self, img, hugs=False):
        # Equalize the image
        img = self.calib.equalize(img)
        self.equilized_img = img

        # Convert to grayscale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        img = cv.Canny(img, 100, 150)

        # Segmentation of the image
        img = self._segmentation(img)

        # Get the hough lines
        if hugs:
            lines = self._getHoughLines(img)
            img = self._drawLines(self.equilized_img, lines)


        return img

    def _segmentation(self, img):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)

        match_mask_color = 255
        
        # Fill inside the polygon
        vertices = self._getVerticesROI(img)
        cv.fillPoly(mask, np.array([vertices], np.int32), match_mask_color)
        
        # Returning the image only where mask pixels match
        masked_image = cv.bitwise_and(img, mask)
        return masked_image

    def _getVerticesROI(self, img):
        # Generate the region of interest
        dim = img.shape
        height = dim[0]
        width = dim[1]
        roi = [
            (0, height - 75),
            (width / 2, (height + 100) / 2),
            (width, height - 75),
        ]

        return roi

    def _drawLines(self, img, lines, color=[0,0,255], thickness=10):
        # If there are no lines to draw, exit.
        if lines is None:
            return    # Make a copy of the original image.
        img = np.copy(img)    # Create a blank image that matches the original in size.
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )    # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_img, (x1, y1), (x2, y2), color, thickness)    # Merge the image with the lines onto the original.
        img = cv.addWeighted(img, 0.8, line_img, 1.0, 0.0)    # Return the modified image.
        return img

    def _getHoughLines(self, img):
        lines = cv.HoughLinesP(
            img,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=20
        )
        return lines


if __name__ == '__main__':
    # Path to video
    video = "img/Udacity/project_video.mp4"
    videoHarder = "img/Udacity/challenge_video.mp4"
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    
    # Start the program
    main = Main(video)
    main1 = Main(videoHarder)
    main2 = Main(videoHardest)

    main.startVideo()
    main.startVideo(True)
    main1.startVideo()
    main1.startVideo(True)
    main2.startVideo()
    main2.startVideo(True)
    