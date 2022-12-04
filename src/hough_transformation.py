import cv2 as cv
import numpy as np
import time
import json
import os

import calib as cal
import preprocess as pre

class HoughTransformation():

    def __init__(self, debug = False) -> None:
        """Constructor of the class HoughTransformation

        Args:
            config_path (str): Path to the config file
            debug (bool, optional): Debug Mode on or off. Defaults to False.
        """
        self.pre = pre.Preprocess()
        self.debug = debug
        self.loaded = False
        
    def load_config(self, path):
        if not os.path.exists(path):
            return print('File '+ path +' not found')
        
        with open(path, 'r') as f:
            config = json.load(f)

        if not config:
            return 'Error: Config file is empty'
        if not 'HOUGH' in config.keys():
            return 'Error: HOUGH is missing'
        if not 'CANNY_LOWER' in config['HOUGH'].keys():
            return 'Error: CANNY_LOWER is missing'
        if not 'CANNY_UPPER' in config['HOUGH'].keys():
            return 'Error: CANNY_UPPER is missing'
        if not 'ROI' in config['HOUGH'].keys():
            return 'Error: ROI is missing'
        if not 'RHO' in config['HOUGH'].keys():
            return 'Error: RHO is missing'
        if not 'THETA' in config['HOUGH'].keys():
            return 'Error: THETA is missing'
        if not 'THRESHOLD' in config['HOUGH'].keys():
            return 'Error: THRESHOLD is missing'
        if not 'MIN_LINE_LENGTH' in config['HOUGH'].keys():
            return 'Error: MIN_LINE_LENGTH is missing'
        if not 'MAX_LINE_GAP' in config['HOUGH'].keys():
            return 'Error: MAX_LINE_GAP is missing'
        if not 'LEFT_FIX' in config['HOUGH'].keys():
            return 'Error: LEFT_FIX is missing'
        if not 'RIGHT_FIX' in config['HOUGH'].keys():
            return 'Error: RIGHT_FIX is missing'
        if not 'BORDER_LEFT' in config['HOUGH'].keys():
            return 'Error: BORDER_LEFT is missing'
        if not 'BORDER_RIGHT' in config['HOUGH'].keys():
            return 'Error: BORDER_RIGHT is missing'
        if not 'POLY_HEIGHT' in config['HOUGH'].keys():
            return 'Error: POLY_HEIGHT is missing'
        
        self.canny_lower = config['HOUGH']['CANNY_LOWER']
        self.canny_upper = config['HOUGH']['CANNY_UPPER']
        self.roi = config['HOUGH']['ROI']
        self.rho = config['HOUGH']['RHO']
        self.theta = config['HOUGH']['THETA']
        self.threshold = config['HOUGH']['THRESHOLD']
        self.min_line_length = config['HOUGH']['MIN_LINE_LENGTH']
        self.max_line_gap = config['HOUGH']['MAX_LINE_GAP']
        self.left_fix = config['HOUGH']['LEFT_FIX']
        self.right_fix = config['HOUGH']['RIGHT_FIX']
        self.border_left = config['HOUGH']['BORDER_LEFT']
        self.border_right = config['HOUGH']['BORDER_RIGHT']
        self.poly_height = config['HOUGH']['POLY_HEIGHT']
        
        
        self.loaded = True
        return None
        

    def execute(self, img):
        """Execute the Hough transformation with it pre-processing steps

        Args:
            img (Img/Frame): Is the frame of the video
            
        Returns:
            Image: The current frame
        """
        
        if not self.loaded:
            return False
        
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
            left_line = self._get_polyLine_points(img, left_x, left_y, self.left_fix, self.border_left)

        if len(right_x) > 0 and len(right_y) > 0:
            right_line = self._get_polyLine_points(img, right_x , right_y, self.right_fix, self.border_right)
        
        if left_line:
            processed_img = self._draw_poly_line_hugh(img, left_line, (255,0,0))
        if right_line:
            processed_img = self._draw_poly_line_hugh(img, right_line)
            
        return processed_img


    def debug_video(self, path, config_path):
        """Video for debugging the video itself

        Args:
            path (String): File path of the video

        Returns:
            Optional (String): Error Description
        """
        if not self.debug:
            return print('Debug mode deactivated, passing...')
        
        error = self.load_config(config_path)
        if error:
            print(error)
            return 

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
            if (type(frame) == bool and not frame) or not frame.any():
                return print('Error: Module not loaded')
            
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
        if self.debug:
            cv.imshow("Debug: Segmentation", img)
        
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
        plot_y = np.linspace(self.poly_height, img.shape[0] - 1, img.shape[0])
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
    videoHarder = "img/Udacity/challenge_video.mp4"
    videoHardest = "img/Udacity/harder_challenge_video.mp4"

    hough_transform = HoughTransformation( debug=True)
    # hough_transform.debug_video(video, "./config/video.json")
    hough_transform.debug_video(videoHarder, "./config/video_challenge.json")
    # hough_transform.debug_video(videoHarder)
    # hough_transform.debug_video(videoHardest)
