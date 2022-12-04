import cv2 as cv
import numpy as np
import time

import calib as cal
import preprocess as pre

class HoughTransformation():
    # Hough Settings
    RHO = 6
    THETA = 90
    THRESHOLD = 100
    MIN_LINE_LENGTH = 3
    MAX_LINE_GAP = 2


    # Canny settings
    CANNY_LOWER = 100
    CANNY_UPPER = 150

    # FIX Point Hough
    LEFT_FIX = (200, 720)
    RIGHT_FIX = (1200, 720)

    def __init__(self, roi, canny_lower = CANNY_LOWER, canny_upper = CANNY_UPPER, rho = RHO, theta = THETA, threshold = THRESHOLD, min_line_length = MIN_LINE_LENGTH, 
    max_line_gap = MAX_LINE_GAP, debug = False, left_fix = LEFT_FIX, right_fix = RIGHT_FIX) -> None:
        """Constructor of the class HoughTransformation

        Args:
            roi (Array): Region of interest for the mask (Rectangle)
            canny_lower (int, optional): Lower border for canny edge detection. Defaults to CANNY_LOWER.
            canny_upper (int, optional): Upper border for the canny
            edge detection. Defaults to CANNY_UPPER.
            rho (int, optional): Rho value for the Hough transformation. Defaults to RHO.
            theta (int, optional): Theta value for the Hough transformation. Defaults to THETA.
            threshold (_type_, optional): Threshold value for the Hough transformation. Defaults to THRESHOLD.
            min_line_length (_type_, optional): Min Line Length value for the Hough transformation. Defaults to MIN_LINE_LENGTH.
            max_line_gap (_type_, optional): Max Line Gap value for the Hough transformation. Defaults to MAX_LINE_GAP.
            debug (bool, optional): Debug Mode on or off. Defaults to False.
            left_fix (_type_, optional): Left Fix point for the line generation. Defaults to LEFT_FIX.
            right_fix (_type_, optional): Right Fix point for the line generation. Defaults to RIGHT_FIX.
        """
        self.pre = pre.Preprocess()
        self.debug = debug
        
        # Preprocess
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.roi = roi

        # Hough
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # Point Hough
        self.left_fix = left_fix
        self.right_fix = right_fix


    def execute(self, img):
        """Execute the Hough transformation with it pre-processing steps

        Args:
            img (Img/Frame): Is the frame of the video
            
        Returns:
            Image: The current frame
        """
        
        processed_img = self._preprocess(img)

        # Hough transformation
        lines = self._getHoughLines(processed_img)
        
        if self.debug: 
            img_debug = self._drawLines(img, lines)
            cv.imshow("Debug: Hugh", img_debug)
        
        # Group the lines into a left and a right group
        line_info = self._group_line_points(img, lines)
        
        # Map line infos
        left_x = line_info['LEFT_X']
        left_y = line_info['LEFT_Y']
        right_x = line_info['RIGHT_X']
        right_y = line_info['RIGHT_Y']
        
        left_line = None
        right_line = None
        
        if len(left_x) > 0 and len(left_y) > 0:
            left_line = self._get_polyLine_points(img, left_x, left_y, self.left_fix, 500)

        if len(right_x) > 0 and len(right_y) > 0:
            right_line = self._get_polyLine_points(img, right_x , right_y, self.right_fix, 610)
        
        if left_line:
            processed_img = self._draw_poly_line_hugh(img, left_line, (255,0,0))
        if right_line:
            processed_img = self._draw_poly_line_hugh(img, right_line)
            
        return processed_img


    def debug_video(self, path):
        """Video for debugging the video itself

        Args:
            path (String): File path of the video

        Returns:
            Optional (String): Error Description
        """
        if not self.debug:
            return print('Debug mode deactivated, passing...')

        calib = cal.Calibration(debug=False)

        # Load video
        video = cv.VideoCapture(path)
        prev_frame_time = 0
        new_frame_time = 0

        # Window size
        win_x = 1280
        win_y = 720
        
        while(video.isOpened()):
            ret, frame = video.read()

            # Break if video is finish or no input
            if not ret:
                break
            
            # Do here the image processing
            frame = cv.resize(frame, (win_x, win_y))
            frame = calib.equalize(frame)

            # Test of the module
            frame = self.execute(frame)

            # Do operations on the frame
            font = cv.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()

            # Calculate Frame Rate
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)

            # Put fps on the screen
            cv.putText(frame, fps, (7, 21), font, 1, (100, 100, 100), 2, cv.LINE_AA)

            cv.imshow('Video', frame)

            if cv.waitKey(1) & 0xFF == ord('p'):
                if cv.waitKey(-1) & 0xFF == ord('p'):
                    next
                else:
                    break
            
        # Stop video and close window
        video.release()
        cv.destroyAllWindows()

    def _preprocess(self, img):
        """Preprocessing steps for the hough transformation

        Args:
            img (Img/Frame): Is the current frame of the video
        
        Returns:
            Image: The current frame
        """
        # Convert to grayscale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Decrease noise via Gauss
        img = self.pre.gauss(img)
        
        # Apply Canny edge detection
        img = self.pre.canny(img, self.canny_lower, self.canny_upper)

        # Segmentation of the image
        img = self.pre.segmentation(img, self.roi)
        
        return img


    def _getHoughLines(self, img):
        """Find the lines in the image via Hough transformation

        Args:
            img (Img/Frame): Current Frame

        Returns:
            Array: Coordinates of the lines
        """
        lines = cv.HoughLinesP(
            img,
            rho=self.rho,
            theta=np.pi / self.theta,
            threshold=self.threshold,
            lines=np.array([]),
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        return lines

    def _drawLines(self, img, lines, color=[0,0,255], thickness=10):
        """Draws lines by given coordinates into an image

        Args:
            img (Image): Current Frame
            lines (Array): Coordinates of the lines
            color (list, optional): Color of the lines. Defaults to [0,0,255].
            thickness (int, optional): Thickness of the lines. Defaults to 10.

        Returns:
            Image: Image with the the lines
        """        
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

    def _group_line_points(self, img, lines):
        """Groups the given line coordinates for the left and right
        part of the image

        Args:
            img (Image): Current Frame
            lines (Array): Line coordinates
        """
         # Get the mid of the picture
        mid = img.shape[1]//2

        left_x = []
        left_y = []

        right_x = []
        right_y = []
        
        # Checks if there are any lines
        if lines is None:
            return {
                'LEFT_X': left_x,
                'LEFT_Y': left_y,
                'RIGHT_X': right_x,
                'RIGHT_Y': right_y,
            }

        factor = 0.06

        for line in lines:
            # Everything 10% left of the mid
            if line[0][0] <= mid - (mid * factor):
                left_x.append(line[0][0])
                left_y.append(line[0][1])
            elif line[0][0] >= mid + (mid * factor):
                right_x.append(line[0][0])
                right_y.append(line[0][1])

            if line[0][2] <= mid - (mid * factor):
                left_x.append(line[0][2])
                left_y.append(line[0][3])
            elif line[0][0] >= mid + (mid * factor):
                right_x.append(line[0][2])
                right_y.append(line[0][3])
        
        return {
            'LEFT_X': left_x,
            'LEFT_Y': left_y,
            'RIGHT_X': right_x,
            'RIGHT_Y': right_y,
        }
    
    def _get_polyLine_points(self, img, x, y, fix_point, border):
        """Generates the polygon fitting the coordinates

        Args:
            img (Image): Current frame
            x (List): x-coordinates of the points
            y (List): y-coordinates of the points
            fix_point (List): Coordinates of an additional fix points
            border (Int): Border of drawing the line (horizon)

        Returns:
            Dict: Info with the fitting x-coordinates (FIT_X) and
            y-coordinates (PLOT_Y)
        """
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
        
        return {
            'FIT_X': fit_x,
            'PLOT_Y': plot_y,
        }    

    def _draw_poly_line_hugh(self, img, draw_info, color = (0,0,255)):
        """Draw the polynomial in the image

        Args:
            img (Image): Current Frame
            draw_info (List): Coordinates of the points to draw
            color (tuple, optional): Color of the line. Defaults to (0,0,255).

        Returns:
            Image: Processed frame
        """
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
    video = "img/Udacity/project_video.mp4"
    roi_video = [
            (300, - 75), # BOTTOM LEFT
            (-55, 72), # TOP LEFT
            (78, 72), # TOP RIGHT
            (-150, - 75), # BOTTOM RIGHT
    ]
    videoHarder = "img/Udacity/challenge_video.mp4"
    videoHardest = "img/Udacity/harder_challenge_video.mp4"

    hough_transform = HoughTransformation(roi_video, debug=True)
    hough_transform.debug_video(video)
    # hough_transform.debug_video(videoHarder)
    # hough_transform.debug_video(videoHardest)
