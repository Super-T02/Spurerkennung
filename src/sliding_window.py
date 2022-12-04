import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time

import calib as cal
import perspective_transform as per
import preprocess as pre

class SlidingWindow():
    
    # Configuration
    N_WINDOWS = 10
    MARGIN = 80
    MIN_PIX = 20
    THRESH = (130, 255)
    LANE_WIDTH_FOR_SEARCH = 10
    SCALING_OF_BOX_WIDTH = 5


    def __init__(self, thresh = THRESH, n_windows = N_WINDOWS, margin = MARGIN, min_pix = MIN_PIX, lane_width_for_search = LANE_WIDTH_FOR_SEARCH,
                 scaling_of_box_width = SCALING_OF_BOX_WIDTH, debug = False, debug_plots = False) -> None:
        """Constructor for the SlidingWindow class

        Args:
            thresh (tuple, optional): Threshold for preprocessing . Defaults to THRESH.
            n_windows (int, optional): Number of sliding windows. Defaults to N_WINDOWS.
            margin (int, optional): margin of the sliding window. Defaults to MARGIN.
            min_pix (int, optional): Minimal number of pixels to
            detect a line. Defaults to MIN_PIX.
            lane_width_for_search (int, optional): Margin for
            searching a line in the frame. Defaults to LANE_WIDTH_FOR_SEARCH.
            scaling_of_box_width (int, optional): Scale of the next window. Defaults to SCALING_OF_BOX_WIDTH.
            debug (bool, optional): Debug mode. Defaults to False.
            debug_plots (bool, optional): Debug mode with plot of histogram. Defaults to False.
        """
        
        self.transformation = per.Transformation(debug)
        self.pre = pre.Preprocess()
        self.debug = debug
        self.debug_plots = debug_plots
        
        # Remember last frames and draw information
        self.last_draw_info = None
        self.last_frame_right_x = None
        self.last_frame_left_x = None
        
        # Configuration
        self.thresh = thresh
        self.n_windows = n_windows
        self.margin = margin
        self.min_pix = min_pix
        self. land_width_for_search = lane_width_for_search
        self.scaling_of_box_width = scaling_of_box_width


    def execute(self, img):
        # Preprocess the image
        img_transformed, M_reverse = self._preprocess(img)

        if self.debug:
            cv.imshow('transformed', img_transformed)
                
        # Set local vars
        hist = self.get_histogram(img_transformed)
        img_y_shape = img_transformed.shape[0]
        img_x_shape = img_transformed.shape[1]
        self.set_context(img_transformed, hist)

        # Generate the windows
        for i_window in range(self.n_windows):
            self._generate_window(img_y_shape, i_window)

        # Get the drawing information
        draw_info = self._generate_line_coordinates(img_y_shape, img_x_shape)

        if not draw_info:

            if not self.last_draw_info:
                return img
            else:
                draw_info = self.last_draw_info
        
        else:
            self.last_draw_info = draw_info

        # Draw the line
        img = self._draw_lane_area(img, img_transformed, M_reverse, draw_info)

        # Return finished frame
        return img


    def _preprocess(self, img):
        """Preprocess the image, apply filters and transformations

        Args:
            img (Image): Current Frame

        Returns:
            Tuple: Transformed image and the reversed transformation matrix
        """
        # Convert to grayscale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        # Apply gaussian blur
        img = self.pre.gauss(img)
        
        # Threshold the image to areal view
        img_transformed, M_reversed = self.transformation.transform_image_perspective(img)
        
        # Apply threshold
        img_transformed = self.pre.threshold(img_transformed, self.thresh)

        return img_transformed, M_reversed


    def get_histogram(self, img):
        """Generate the histogram of the transformed image

        Args:
            img (Image): Current Frame

        Returns:
            List: Histogram of the image
        """
        # Get the histogram of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

        # Show the histogram in debug mode
        if self.debug_plots:
            plt.figure()
            plt.plot(histogram)
            plt.show()

        return histogram
    

    def set_context(self, img, hist):
        """Generate the search context for the windows (based on the previous frame)

        Args:
            img (Image): Current Frame
            hist (List): Histogram of the current frame
        """
        # Generate black layered image
        if self.debug: self.debug_img = np.dstack((img, img, img)) * 255
        mid = img.shape[1]//2

        if not self.last_frame_left_x:
            # Divide the histogram into to parts
            leftx_base = np.argmax(hist[:mid])
        else:
            left_negative = self.last_frame_left_x - self.land_width_for_search
            left_positive = self.last_frame_left_x + self.land_width_for_search
            if left_negative < 1:
                left_negative = 1
                left_positive = mid
            leftx_base = np.argmax(hist[left_negative : left_positive]) + left_negative
            self.last_frame_left_x = None
            
        if not self.last_frame_right_x:
            # Divide the histogram into to parts
            rightx_base = np.argmax(hist[mid:]) + mid
        else:
            right_negative = self.last_frame_right_x - self.land_width_for_search
            right_positive = self.last_frame_right_x + self.land_width_for_search
            if right_positive > img.shape[1] - 1:
                right_negative = mid
                right_positive = img.shape[1] - 1 
            rightx_base = np.argmax(hist[right_negative : right_positive]) + right_negative
            self.last_frame_right_x = None


        # Number of sliding windows in the frame
        # self.n_windows = 10
        self.window_height = img.shape[0]//self.n_windows

        # Find coordinates which are not zero
        nonzero = img.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        self.current_leftx = leftx_base
        self.current_rightx = rightx_base

        self.left_lane_inds = []
        self.right_lane_inds = []


    def _generate_window(self, img_y_shape, index):
        """Generate the window for the current index

        Args:
            img_y_shape (int): y shape of the image
            index (int): index of the current window
        """
        # Set current window coordinates
        win_y_low = img_y_shape - (index + 1) * self.window_height
        win_y_high = img_y_shape - index * self.window_height
        
        # Define box-window coordinates
        win_xleft_low = self.current_leftx - (self.margin + index * self.scaling_of_box_width)
        win_xleft_high = self.current_leftx + (self.margin + index * self.scaling_of_box_width)
        win_xright_low = self.current_rightx - (self.margin + index * self.scaling_of_box_width)
        win_xright_high = self.current_rightx + (self.margin + index * self.scaling_of_box_width)

        # Define rectangle
        if self.debug:
            if index == 0: 
                cv.rectangle(self.debug_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 0), 2)
                cv.rectangle(self.debug_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 0), 2)
            else:
                cv.rectangle(self.debug_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv.rectangle(self.debug_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Get the indices where the coordinates of the image are not
        # zero but in the window (defined by the win_y_low, ...)
        left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
        right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
        self.left_lane_inds.append(left_inds)
        self.right_lane_inds.append(right_inds)

        # Change the current indices
        if len(left_inds) > self.min_pix:
            self.current_leftx = int(np.mean(self.nonzerox[left_inds]))
            if index == 0:
                self.last_frame_left_x = self.current_leftx
        if len(right_inds) > self.min_pix:
            self.current_rightx = int(np.mean(self.nonzerox[right_inds]))
            if index == 0:
                self.last_frame_right_x = self.current_rightx

        # Save x coordinate of first window from this frame for next frame to search here for lane (doesn't jump around)
        # if index == self.n_windows - 1:
        #     self.last_frame_left_x = self.current_leftx
        #     self.last_frame_right_x = self.current_rightx


    def _generate_line_coordinates(self, img_y_shape, img_x_shape):
        """Generate the line coordinates for the left and right lane

        Args:
            img_y_shape (int): shape of the image in y direction
            img_x_shape (int): shape of the image in x direction

        Returns:
            Dict: Dictionary with the left and right lane coordinates
        """
        # Flatten array
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        # Check wether lines are detected:
        if len(leftx) <= 0 or len(lefty) <= 0 or len(rightx) <= 0 or len(righty) <= 0:
            # Prepare return values
            ret = {}
            return ret

        # Create line
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Show plot of the created lines
        plot_y = np.linspace(0, img_y_shape - 1, img_y_shape)
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

        if self.debug:
            # Prepared layered image
            self.debug_img[lefty, leftx] = [255, 0, 0]
            self.debug_img[righty, rightx] = [0, 0, 255]

            coord_left = np.array(np.dstack((left_fit_x, plot_y))[0], np.int)
            coord_right = np.array(np.dstack((right_fit_x, plot_y))[0], np.int)

            cv.polylines(self.debug_img, [coord_left], False, (0,255,255), thickness=3)
            cv.polylines(self.debug_img, [coord_right], False, (0,255,255), thickness=3)

            cv.imshow('Sliding Window', self.debug_img)

        # Prepare return values
        return {
            'LEFT_X': leftx,
            'RIGHT_X': rightx,
            'LEFT_FIT_X': left_fit_x,
            'RIGHT_FIT_X': right_fit_x,
            'PLOT_Y': plot_y
        }

    def _draw_lane_area(self, original_img, transformed_img, M_reversed, draw_info):
        """Draw the lane area on the original image

        Args:
            original_img (Image): original image
            transformed_img (Image): transformed image
            M_reversed (List): reversed transformation matrix
            draw_info (Dict): dictionary with the coordinates of the lane

        Returns:
            Image: image with the lane area
        """
        # Unpack draw Info
        left_fit_x = draw_info['LEFT_FIT_X']
        right_fit_x = draw_info['RIGHT_FIT_X']
        plot_y = draw_info['PLOT_Y']

        transformed_zero = np.zeros_like(transformed_img).astype(np.uint8)
        color_transformed = np.dstack((transformed_zero, transformed_zero, transformed_zero))

        # Generate the points of the lane
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the driving lane in the transformed image
        cv.fillPoly(color_transformed, np.int_([pts]), (0,255,0))

        # Untransform the image back 
        new_transformed = cv.warpPerspective(color_transformed, M_reversed, (original_img.shape[1], original_img.shape[0]))
        result = cv.addWeighted(original_img, 1, new_transformed, 0.3, 0)

        # Show the transformed aria
        if self.debug: cv.imshow("Transformed Aria", new_transformed)

        return result

    def debug_video(self, path):
        """Debug the module with a video

        Args:
            path (String): path to the video

        Returns:
            String: Error message
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

if __name__ == '__main__':
    video = "img/Udacity/project_video.mp4"
    videoHarder = "img/Udacity/challenge_video.mp4"
    videoHardest = "img/Udacity/harder_challenge_video.mp4"

    slide_win = SlidingWindow((140,255), debug=True)
    slide_win.debug_video(video)
    slide_win.debug_video(videoHarder)
    slide_win.debug_video(videoHardest)
