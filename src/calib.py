import cv2 as cv
import numpy as np
import glob


class Calibration():
    # Chessboard Configuration
    GRID_SIZE = (9, 6)
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, debug: bool) -> None:
        self.debug = debug

        # Arrays to store object points and image points from all the images.
        self.objPoints = [] # 3d point in real world space
        self.imgPoints = [] # 2d points in image plane.

        self._chessboard()

    def _chessboard(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.GRID_SIZE[0] * self.GRID_SIZE[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.GRID_SIZE[0],0:self.GRID_SIZE[1]].T.reshape(-1,2)

        # read the chessboard imag
        images = glob.glob('./img/Udacity/calib/*.jpg')

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
                    cv.imshow('img', board)
                    if cv.waitKey(500) & 0xFF == ord('q'):
                        break

        if self.debug: cv.destroyAllWindows()

if __name__ == '__main__':
    Calibration(debug=True)