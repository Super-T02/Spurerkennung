import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import pickle
import glob
import os


class Calibration():
    # Chessboard Configuration
    GRID_SIZE = (9, 6)
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    SAVE_PATH = 'calibration.calib'
    CALIB_PATH = 'img/Udacity/calib'

    def __init__(self, debug: bool, file_path: str = None) -> None:
        self.debug = debug
        self.calibrated = False

        # Initialize the calibration matrix and distortion coefficients
        self.mtx, self.dist, self.roi, self.newcameramtx = None, None, None, None
        self.rvecs, self.tvecs = None, None

        # Arrays to store object points and image points from all the images.
        self.objPoints = [] # 3d point in real world space
        self.imgPoints = [] # 2d points in image plane.

        # Load calibration if file provided or the calibration already exists
        if file_path and os.path.exists(file_path): self.loadCalibration(file_path)
        elif os.path.exists(self.SAVE_PATH): self.loadCalibration(self.SAVE_PATH)

    def loadCalibration(self, file_path):
        with open(file_path, 'rb') as f:
            self.mtx, self.dist, self.roi, self.newcameramtx = pickle.load(f)
        self.calibrated = True

    def saveCalibration(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.mtx, self.dist, self.roi, self.newcameramtx), f)

    def equalize(self, img):
        if not self.calibrated: self._calibrate(img)

        # undistort the image
        undisortedImage = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

        # crop the image
        x, y, w, h = self.roi
        undisortedImage = undisortedImage[y:y+h, x:x+w]
        return undisortedImage

    def _chessboard(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.GRID_SIZE[0] * self.GRID_SIZE[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.GRID_SIZE[0],0:self.GRID_SIZE[1]].T.reshape(-1,2)

        # read the chessboard imag
        images = glob.glob(self.CALIB_PATH + '/*.jpg')

        for file_name in images:
            board = cv.cvtColor(cv.imread(file_name), cv.COLOR_BGR2RGB)
            gray_board = cv.cvtColor(board, cv.COLOR_RGB2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray_board, self.GRID_SIZE, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objPoints.append(objp)
                corners2 = cv.cornerSubPix(gray_board,corners, (11,11), (-1,-1), self.CRITERIA)
                self.imgPoints.append(corners)

                # Draw and display the corners
                if self.debug: 
                    cv.drawChessboardCorners(board, self.GRID_SIZE, corners2, ret)
                    print("Grid matched: " + file_name)
                    cv.imshow('img', board)
                    if cv.waitKey(100) & 0xFF == ord('q'):
                        break

        if self.debug: cv.destroyAllWindows()

    def _calibrate(self, img):
        self._chessboard()
        self._generateMatrix(img)
        self.saveCalibration(self.SAVE_PATH)        

    def _generateMatrix(self, img):
        # Generate the calibration matrix
        img_cvt = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objPoints, self.imgPoints, img_cvt.shape[::-1], None, None)
        
        img = cv.imread('img/Udacity/calib/calibration3.jpg')
        h,  w = img.shape[:2]
        self.newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        self.calibrated = True

if __name__ == '__main__':
    calib = Calibration(debug=True)
    plt.imshow(calib.equalize(cv.imread('img/Udacity/calib/calibration3.jpg', cv.COLOR_BGR2RGB)))
    plt.show()