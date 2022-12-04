import cv2 as cv
import os
import time

import calib as cal
import sliding_window as slw
import hough_transformation as hou

class Main():
    # Window size
    WIN_X = 1280
    WIN_Y = 720

    def __init__(self, path, debug=False):
        print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')
                
        # Define the objects
        self.calib = cal.Calibration(debug=debug)
        self.sliding_win = slw.SlidingWindow(debug=debug)
        self.hough = hou.HoughTransformation(debug = debug)
        
        # Define the variables
        self.path = path
        
    def startVideo(self, mode=0, config_path="./config/video.json"):
        if not os.path.exists(self.path):
            return print('Video not found')
        
        # Load config
        error = None
        if mode==0:
            error = self.hough.load_config(config_path)
        elif mode == 1:
            error = self.sliding_win.load_config(config_path)
        else:
            error = "Mode not found"
        
        if error:
            print(error)
            return 

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
            
            if (type(frame) == bool and not frame) or not frame.any():
                return print('Error: Module not loaded')

            # Do operations on the frame
            if frame is not None:
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
    videoHarder = "img/Udacity/challenge_video.mp4"
    # roi_videoHarder = [
    #         (300, - 75),
    #         (-40, 90),
    #         (50, 80),
    #         (-325, - 75),
    # ]
    videoHardest = "img/Udacity/harder_challenge_video.mp4"
    
    # Start the program
    main = Main(video, debug=False)

    # Mode:
    # - 0: Hough
    # - 1: Sliding window
    main.startVideo(mode=0, config_path="./config/video.json")
    main.startVideo(mode=1, config_path="./config/video.json")
    