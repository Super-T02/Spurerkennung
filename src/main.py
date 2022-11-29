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
            self.equilized_img = frame
            if mode == 1:
                frame = self.sliding_win.apply_sliding_window(frame)
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

    def _preprocess_default(self, img, hough=False):
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
            if self.debug: 
                img_debug = self._drawLines(self.equilized_img, lines)
                cv.imshow("Debug: Hugh", img_debug)
            line_info = self._group_line_points(self.equilized_img, lines)
            left_x = line_info['LEFT_X']
            left_y = line_info['LEFT_Y']
            right_x = line_info['RIGHT_X']
            right_y = line_info['RIGHT_Y']
            
            if len(left_x) > 0 and len(left_y) > 0:
                self.left_line = self._get_polyLine_Points(self.equilized_img, left_x, left_y, self.LEFT_FIX, 500)

            if len(right_x) > 0 and len(right_y) > 0:
                self.right_line = self._get_polyLine_Points(self.equilized_img, right_x , right_y, self.RIGHT_FIX, 610)
            
            img = self._draw_poly_line_hugh(self.equilized_img, self.left_line)
            img = self._draw_poly_line_hugh(self.equilized_img, self.right_line)

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
            ((width / 2) + self.roi[1][0], (height / 2) + self.roi[1][1]),
            ((width / 2) + self.roi[2][0], (height / 2) + self.roi[2][1]),
            (width + self.roi[3][0], height + self.roi[3][1]),
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

    def _group_line_points(self, img, lines):
         # Get the mid of the picture
        mid = img.shape[1]//2

        left_x = []
        left_y = []

        right_x = []
        right_y = []
        
        if lines is None:
            ret = {
                'LEFT_X': left_x,
                'LEFT_Y': left_y,
                'RIGHT_X': right_x,
                'RIGHT_Y': right_y,
            }
            return ret

        for line in lines:
            if line[0][0] <= mid:
                left_x.append(line[0][0])
                left_y.append(line[0][1])
            else:
                right_x.append(line[0][0])
                right_y.append(line[0][1])

            if line[0][2] <= mid:
                left_x.append(line[0][2])
                left_y.append(line[0][3])
            else:
                right_x.append(line[0][2])
                right_y.append(line[0][3])
        
        ret = {
            'LEFT_X': left_x,
            'LEFT_Y': left_y,
            'RIGHT_X': right_x,
            'RIGHT_Y': right_y,
        }
        
        return ret

    def _get_polyLine_Points(self, img, x, y, fix_point, border):
        # Add point of car if the nearest point is further away then the
        # provided value
        if y[np.argmax(y)] < border:
            x.append(fix_point[0])
            y.append(fix_point[1])

        #Generate poly lines
        poly = np.polyfit(y,x,2)

        # Generate the points
        plot_y = np.linspace(385, img.shape[0] - 1, img.shape[0])
        fit_x = poly[0] * plot_y**2 + poly[1] * plot_y + poly[2]

        info = {
            'FIT_X': fit_x,
            'PLOT_Y': plot_y,
        }

        return info

    def _draw_poly_line_hugh(self, img, draw_info, color = (0,0,255)):
        # Unpack draw Info
        fit_x = draw_info['FIT_X']
        plot_y = draw_info['PLOT_Y']

        # Check whether data exist
        if len(fit_x) <= 0 and len(plot_y) <= 0:
            return img

        # Generate the points of the lane
        pts = np.array(np.transpose(np.vstack([fit_x, plot_y])))

        pts = pts.reshape((-1, 1, 2))

        # Draw the driving lane in the transformed image
        cv.polylines(img, np.int_([pts]), False, color, 4)

        return img

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
            (-60, 70),
            (40, 70),
            (-150, - 75),
    ]
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    roi_videoHardest = [
            (300, - 75),
            (-60, 70),
            (40, 70),
            (-150, - 75),
    ]
    
    # Start the program
    main = Main(video, roi_video, 1, debug=False) # canny_lower=50, canny_upper=100 if you change the order of areal view preprocessing 
    main1 = Main(videoHarder, roi_videoHarder, canny_lower=15, canny_upper=30)
    main2 = Main(videoHardest, roi_videoHardest)

    # Mode:
    # - 0: Hough
    # - 1: Sliding window

    main.startVideo()
    main.startVideo(mode=1)
    main.startVideo(hough=True, show_areal=True)
    main.startVideo(hough=True)
    # main1.startVideo()
    # main1.startVideo(hough=True)
    # main2.startVideo()
    # main2.startVideo(hough=True)
    