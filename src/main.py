import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
import time

import calib as cal
import perspective_transform as per

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
    HOUG_RHO = 2
    HOUG_THETA = 180
    HOUGH_THRESHOLD = 120
    HOUGH_MIN_LINE_LENGTH = 40
    HOUGH_MAX_LINE_GAP = 5


    def __init__(self, path, roi, canny_lower = None, canny_upper = None):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')
        if path is None or roi is None: return print('No path or roi given')
        self.calib = cal.Calibration(False)
        self.transformation = per.Transformation(False)
        self.path = path
        self.roi = roi
        self.canny_lower = canny_lower if canny_lower is not None else self.CANNY_LOWER
        self.canny_upper = canny_upper if canny_upper is not None else self.CANNY_UPPER
        
    def startVideo(self, hough=False, show_areal=False):
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
            frame = self._preprocess_default(frame, hough=hough) if not show_areal else self._preprocess_areal_view(frame)

            # Do operations on the frame
            if frame is not None:
                frame = cv.resize(frame, (self.WIN_X, self.WIN_Y))
                
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
        img = self.transformation.transform_image_perspective(img)
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

    def _preprocess_default(self, img, hough=False):
        # Equalize the image
        img = self.calib.equalize(img)
        self.equilized_img = img

        # Convert to grayscale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Decrease noise via gauss
        img = self._gauss(img)

        # Apply Canny edge detection
        img = self._canny(img)

        # Segmentation of the image
        img = self._segmentation(img)

        # Get the hough lines
        if hough:
            lines = self._getHoughLines(img)
            img = self._drawLines(self.equilized_img, lines)


        return img

    def _gauss(self, img):
        return cv.GaussianBlur(img, self.GAUSS_KERNEL, 0)

    def _canny(self, img):
        return cv.Canny(img, self.canny_lower, self.canny_upper)

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
            (self.roi[0][0], height + self.roi[0][1]),
            ((width + self.roi[1][0]) / 2, (height + self.roi[1][1]) / 2),
            (width + self.roi[2][0], height + self.roi[2][1]),
        ]

        return roi

    def _drawLines(self, img, lines, color=[0,0,255], thickness=10):
        # Make a copy of the original image.
        img = np.copy(img)
        
        # Create a blank image that matches the original in size.
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )

        # Check if any lines were detected
        if lines is not None:

            # Loop over all lines and draw them on the blank image.
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv.line(line_img, (x1, y1), (x2, y2), color, thickness)
        
        # Add the lines to the original image
        img = cv.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        
        # Return the modified image.
        return img

    def _getHoughLines(self, img):
        lines = cv.HoughLinesP(
            img,
            rho=self.HOUG_RHO,
            theta=np.pi / self.HOUG_THETA,
            threshold=self.HOUGH_THRESHOLD,
            lines=np.array([]),
            minLineLength=self.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=self.HOUGH_MAX_LINE_GAP
        )
        return lines


if __name__ == '__main__':
    # Path to video
    video = "img/Udacity/project_video.mp4"
    roi_video = [
            (80, - 75),
            (50, 120),
            (-120, - 75),
    ]
    videoHarder = "img/Udacity/challenge_video.mp4"
    roi_videoHarder = [
            (180, - 75),
            (80, 160),
            (-180, - 75),
    ]
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    roi_videoHardest = [
            (80, - 75),
            (50, 120),
            (-120, - 75),
    ]
    
    # Start the program
    main = Main(video, roi_video) # canny_lower=50, canny_upper=100 if you change the order of areal view preprocessing 
    main1 = Main(videoHarder, roi_videoHarder, canny_lower=15, canny_upper=30)
    main2 = Main(videoHardest, roi_videoHardest)

    main.startVideo()
    main.startVideo(hough=True, show_areal=True)
    main.startVideo(hough=True)
    # main1.startVideo()
    # main1.startVideo(hough=True, show_areal=True)
    # main2.startVideo()
    # main2.startVideo(hough=True)
    