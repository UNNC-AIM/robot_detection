import numpy as np
import cv2
from ctypes import *
from detect_function import YOLOv5Detector
import pupil_apriltags as apriltags

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, vx, vy], position and velocity
        self.state = np.array([0, 0, 0, 0], dtype=float)

        # State transition matrix F
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # Observation matrix H
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Prediction error covariance matrix P
        self.P = np.eye(4) * 1000  # Initial value

        # Observation noise covariance matrix R
        self.R = np.eye(2) * 5  # Observation noise

        # Process noise covariance matrix Q
        self.Q = np.eye(4)

        # Identity matrix I
        self.I = np.eye(4)

    def predict(self):
        # State prediction
        self.state = np.dot(self.F, self.state)
        # Covariance prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]  # Return the predicted [x, y]

    def update(self, z):
        # z is the new observation [x, y]
        z = np.array(z, dtype=float)
        y = z - np.dot(self.H, self.state)  # Observation residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain

        # Update state
        self.state = self.state + np.dot(K, y)
        # Update covariance matrix
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)

        return self.state[:2]  # Return the updated [x, y]

weights_path = 'models/car.engine'
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.3, iou_thres=0.5, max_det=1, ui=True)
cam_caliMatrix = np.load('./cam_calibration.npy')
matrix = cam_caliMatrix['camera_matrix']
dist = cam_caliMatrix['dist_coeffs']
apriltags_detector = apriltags.Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, 
                                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
tag_size = .1 #TODO: modify if needed
object_points = np.array([[-tag_size/2, -tag_size/2, 0],
                          [ tag_size/2, -tag_size/2, 0],
                          [ tag_size/2,  tag_size/2, 0],
                          [-tag_size/2,  tag_size/2, 0]], dtype=np.float32)

def aTag_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    if tags:
        corners = tags[0].corners
        cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 5)
        return corners

def detection(img): #yolo detection
    result = detector.predict(img)
    for detection in result:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # find center point
            x_center = left + w // 2
            y_center = top + h // 2
            print('positive detection')
            return x_center, y_center  
        
    return None, None
def apply_affine_transform(input_pts, output_pts, frame):
    # Calculate the affine transformation matrix
    matrix = cv2.getAffineTransform(np.float32(input_pts), np.float32(output_pts))
    # Apply the affine transformation to the frame
    transformed_frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
    return transformed_frame

def cord_transform(input_pt, src, dst):
    src_points = np.array(src, dtype=np.float32)
    dst_points = np.array(dst, dtype=np.float32)
    
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    input_point = np.array([[input_pt]], dtype=np.float32)

    transformed_point = cv2.transform(input_point, affine_matrix)
    
    x_prime, y_prime = transformed_point[0][0]
    
    return (x_prime, y_prime)

def kalman_filter(coordinates):
    kf = KalmanFilter()
    predictions = []

    for coord in coordinates:
        prediction = kf.predict()  
        updated = kf.update(coord)  # update coordinates using observation
        predictions.append(updated)

    return predictions[-1]

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)


calibration_points_fr1 = []
calibration_points_fr2 = []
output_points = [[0, 0], [640, 0], [640, 480]]
car_coords1 = [[0,0]]
car_coords2 = [[0,0]]

x_values = np.arange(0, 641, 40)
y_values = np.arange(0, 481, 40)
# set up grid points
grid_x, grid_y = np.meshgrid(x_values, y_values)
grid_points = np.stack((grid_x, grid_y), axis=-1)


# Mouse callback function to record the mouse click position
def get_mouse_click1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        print(f"Mouse clicked at position: ({x}, {y})")
        calibration_points_fr1.append([x, y]) # ul ur dr
        print(len(calibration_points_fr1))
def get_mouse_click2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        print(f"Mouse clicked at position: ({x}, {y})")
        calibration_points_fr2.append([x, y]) #same as above
        print(len(calibration_points_fr2))

def no_mouse_click(event, x, y, flags, param): 
    pass


def draw_point_on_frame(frame, point, color=(0, 255, 0), radius=5, thickness=-1):
    # 绘制
    cv2.circle(frame, point, radius, color, thickness)

def nearest_neighbor(grid_points, pt):

    pt = np.array(pt)
    
     # Calculate the Euclidean distance between each grid point and the input coordinates
    distances = np.linalg.norm(grid_points - pt, axis=-1)
    
    # find index
    nearest_index = np.unravel_index(np.argmin(distances), distances.shape)
    
    # return nearest pt
    nearest_point = tuple(grid_points[nearest_index])
    
    return nearest_point

def transform_point(point, R, T):

    point_homogeneous = np.array([point[0], point[1], 0, 1], dtype=np.float32)
    
    # transform
    transformed_point = T @ point_homogeneous[:3] + R
    
    return transformed_point[:2]

def ralative_coord(x,y,R,T):
    # Returns coordinates based on positional estimation   
    homo_pt =  np.array([x,y,0,1], dtype=np.float32)
    transformed_point = T @ homo_pt +R

    return transformed_point[:2]
    

cv2.namedWindow('frame1')
cv2.setMouseCallback('frame1', get_mouse_click1)
cv2.namedWindow('frame2')
cv2.setMouseCallback('frame2', get_mouse_click2)
print("Click on the four corners of the screen in order: top left, top right, bottom right")
cam_pos = None
def process_frame(frame, calibration_points, output_points, frame_number):
    transformed_frame = apply_affine_transform(input_pts=calibration_points, output_pts=output_points, frame=frame)
    transformed_frame = cv2.resize(transformed_frame, (640, 480), interpolation=cv2.INTER_AREA)
    if frame_number == 1:
        car_coords = car_coords1
    if frame_number == 2:
        car_coords = car_coords2
    car_x, car_y = detection(frame)
    if car_x is not None and car_y is not None:
        target_coords = cord_transform((car_x, car_y), calibration_points, output_points)
        draw_point_on_frame(transformed_frame, (int(target_coords[0]), int(target_coords[1])))
        cv2.putText(transformed_frame, "car", (int(target_coords[0]) + 10, int(target_coords[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        car_coords.append([target_coords[0], target_coords[1]])
        nearest_neighbor_coords = nearest_neighbor(grid_points, target_coords)
        draw_point_on_frame(transformed_frame, nearest_neighbor_coords, (0, 0, 255))

        # kalman filter
        filtered_coords = kalman_filter(car_coords)
        draw_point_on_frame(transformed_frame, (int(filtered_coords[0]), int(filtered_coords[1])), (255, 0, 0))
        
        prediction_neighbor = nearest_neighbor(grid_points, filtered_coords)
        draw_point_on_frame(transformed_frame, prediction_neighbor, (255, 255, 255), 2)
        cv2.putText(transformed_frame, 'prediction', (int(filtered_coords[0]) + 10, int(filtered_coords[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return transformed_frame

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)

    if cam_pos is None:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        corners1 = aTag_detection(frame1)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        corners2 = aTag_detection(frame2)
        
        if corners1 and corners2:
            ret1, rvec1, tvec1 = cv2.solvePnP(object_points, corners1, matrix, dist)
            ret2, rvec2, tvec2 = cv2.solvePnP(object_points, corners2, matrix, dist)
            if ret1 and ret2:
                R1, _ = cv2.Rodrigues(rvec1)
                R2, _ = cv2.Rodrigues(rvec2)
                R = R2 @ R1.T
                T = tvec2 - R @ tvec1
                print("rotation matrix: \n", R)
                print("translation vector: \n", T)
                cam_pos = (R, T)

    if len(calibration_points_fr1) == 3:
        cv2.setMouseCallback("frame1", no_mouse_click)
        transformed_frame1 = process_frame(frame1, calibration_points_fr1, output_points, 1)
        cv2.imshow('transform1', transformed_frame1)

    if len(calibration_points_fr2) == 3:
        cv2.setMouseCallback("frame2", no_mouse_click)
        transformed_frame2 = process_frame(frame2, calibration_points_fr2, output_points, 2)
        cv2.imshow('transform2', transformed_frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
