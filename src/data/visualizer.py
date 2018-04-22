import os

import numpy as np
import pydicom as pc
import cv2
import pylab

class Visualizer(object):
    def __init__(self):
        #cv2.namedWindow("figure")
        passc
        img = dcm.pixel_array
        contour = np.array(contour)
        cv2.fillConvexPoly(img, contour, 1)
        pylab.imshow(img, cmap=pylab.cm.bone)
        pylab.show()
        cv2.waitKey(0)
    