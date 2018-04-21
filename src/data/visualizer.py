import os

import numpy as np
import pydicom as pc
import cv2
import pylab

class Visualizer(object):
    def __init__(self):
        #cv2.namedWindow("figure")
        pass

    def visualize(self, filename, contour):
        dcm = pc.read_file(filename)
        print(dcm.SOPInstanceUID)
        img = dcm.pixel_array
        contour = np.array(contour)
        cv2.drawContours(img, contour, -1, (2000,2000,2000), 1)
        pylab.imshow(img, cmap=pylab.cm.bone)
        pylab.show()
        cv2.waitKey(0)
    