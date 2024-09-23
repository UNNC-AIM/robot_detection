import cv2 
import numpy as np
import os
import pupil_apriltags as apriltags


cam_calibration = np.load('./cam_calibration.npy')
matrix = cam_calibration['camera_matrix']
dist = cam_calibration['dist_coeffs']

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
detector = apriltags.Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
tag_size = 0.05 #TODO:视情况修改(m)
object_points = np.array([[-tag_size/2, -tag_size/2, 0],
                          [ tag_size/2, -tag_size/2, 0],
                          [ tag_size/2,  tag_size/2, 0],
                          [-tag_size/2,  tag_size/2, 0]], dtype=np.float32)

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    if tags:
        corners = tags[0].corners
        cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 5)
        return corners
