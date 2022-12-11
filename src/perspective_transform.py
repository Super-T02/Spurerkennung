import time
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from calib import Calibration as cal

class Transformation():

    def __init__(self, debug = False) -> None:
        self.debug = debug

    def _calculate_matrix(self, img, trans_matrix):
        src_coor = trans_matrix['SRC']
        dst_coor = trans_matrix['DST']
        src = np.float32([src_coor[0], src_coor[1], src_coor[2], src_coor[3]]) 
        dst = np.float32([dst_coor[0], dst_coor[1], dst_coor[2], dst_coor[3]])

        M = cv.getPerspectiveTransform(src,dst)
        M_reversed = cv.getPerspectiveTransform(dst,src)
        return M, M_reversed


    def transform_image_perspective(self, img, trans_matrix):
        M, M_reversed = self._calculate_matrix(img, trans_matrix)

        img_transformed = cv.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv.INTER_LINEAR)

        return img_transformed, M_reversed


    def _set_points_in_picture(self, img):
        height = len(img) -1
        width = len(img[0]) -1

        src_coor, _ = self._get_transformation_coordinates(height, width)
        src = np.float32([src_coor[0], src_coor[1], src_coor[2], src_coor[3]])

        for val in src: 
            cv.circle(img,(int(val[0]),int(val[1])),5,(0,255,0),-1)
    
        return img

    
    def debug_mode(self):
        if not self.debug:
            return print('Debug mode deactivated, passing...')

        calib = cal(debug=False)
        # Path to video
        video = "img/Udacity/project_video.mp4"
        videoHarder = "img/Udacity/challenge_video.mp4"
        videoHardest = "img/Udacity/harder_challenge_video.mp4"

        # Load video
        video = cv.VideoCapture(videoHarder)
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

            # Do operations on the frame
            transformed, _ = self.transform_image_perspective(frame)
            frame = self._set_points_in_picture(frame)
            font = cv.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()

            # Calculate Frame Rate
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)

            # Put fps on the screen
            cv.putText(frame, fps, (7, 21), font, 1, (100, 100, 100), 2, cv.LINE_AA)
            cv.putText(transformed, fps, (7, 21), font, 1, (100, 100, 100), 2, cv.LINE_AA)

            cv.imshow('Video', frame)
            cv.imshow('transformed', transformed)

            if cv.waitKey(-1) & 0xFF == ord('n'):
                next
            else:
                break           

        # Stop video and close window
        video.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    transform = Transformation(debug=True)
    transform.debug_mode()
    