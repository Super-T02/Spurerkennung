import cv2 as cv
import numpy as np

class Preprocess():

    GAUSS_KERNEL = (5, 5)
    
    def __init__(self) -> None:
        pass

    def gauss(self, img):
        return cv.GaussianBlur(img, self.GAUSS_KERNEL, 0)

    def canny(self, img, canny_lower, canny_upper):
        return cv.Canny(img, canny_lower, canny_upper)
    
    def threshold(self, img, thresh):
        return cv.threshold(img, thresh[0], thresh[1], cv.THRESH_BINARY)[1]

    
    def segmentation(self, img, roi):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)

        match_mask_color = 255
        
        # Fill inside the polygon
        vertices = self._generateCoordinatesRectangle(img, roi)
        cv.fillPoly(mask, np.array([vertices], np.int32), match_mask_color)
        
        # Returning the image only where mask pixels match
        masked_image = cv.bitwise_and(img, mask)
        return masked_image

    def _generateCoordinatesRectangle(self, img, roi):
        # Generate the region of interest
        dim = img.shape
        height = dim[0]
        width = dim[1]
        roi = [
            (roi[0][0], height + roi[0][1]),
            ((width / 2) + roi[1][0], (height / 2) + roi[1][1]),
            ((width / 2) + roi[2][0], (height / 2) + roi[2][1]),
            (width + roi[3][0], height + roi[3][1]),
        ]

        return roi
    
    def map_color(self, img, lower, upper):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # Threshold of the color in HSV space
        lower = np.array(lower)
        upper = np.array(upper)
        
        # preparing the mask to overlay
        mask = cv.inRange(hsv, lower, upper)
        
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-blue regions
        result = cv.bitwise_and(img, img, mask = mask)
        result = self.threshold(result, (1, 255))
        result = cv.bitwise_or(img, result)
        return result