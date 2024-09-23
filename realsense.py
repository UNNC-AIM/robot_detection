import sys
import threading
import msvcrt
import numpy as np
import cv2
from ctypes import *
from detect_function import YOLOv5Detector
 
class KalmanFilter:
    def __init__(self):
        # 状态向量 [x, y, vx, vy]，位置和速度
        self.state = np.array([0, 0, 0, 0], dtype=float)

        # 状态转移矩阵 F
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # 观测矩阵 H
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # 预测误差协方差矩阵 P
        self.P = np.eye(4) * 1000  # 初始值

        # 观测噪声协方差矩阵 R
        self.R = np.eye(2) * 5  # 观测噪声

        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(4)

        # 单位矩阵 I
        self.I = np.eye(4)

    def predict(self):
        # 状态预测
        self.state = np.dot(self.F, self.state)
        # 协方差预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]  # 返回预测的 [x, y]

    def update(self, z):
        # z 是新的观测 [x, y]
        z = np.array(z, dtype=float)
        y = z - np.dot(self.H, self.state)  # 观测残差
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # 残差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

        # 更新状态
        self.state = self.state + np.dot(K, y)
        # 更新协方差矩阵
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)

        return self.state[:2]  # 返回更新后的 [x, y]
    
weights_path = 'models/car.engine'
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.3, iou_thres=0.5, max_det=1, ui=True)


def detection(img):
    result = detector.predict(img)
    
    for detection in result:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # 计算中心点坐标
            x_center = left + w // 2
            y_center = top + h // 2
            print('positive detection')
            return x_center, y_center  # 找到车辆后直接返回中心点坐标

    # 没有 car返回 None
    return None, None
def apply_affine_transform(input_pts, output_pts, frame):
    # 计算仿射变换矩阵
    matrix = cv2.getAffineTransform(np.float32(input_pts), np.float32(output_pts))
    # 对帧进行仿射变换
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
        prediction = kf.predict()  # 黑魔法
        updated = kf.update(coord)  # 用观测更新坐标
        predictions.append(updated)

    return predictions[-1]

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)


calibration_points_fr1 = []
calibration_points_fr2 = []
output_points = [[0, 0], [640, 0], [640, 480]]
car_cords1 = [[0,0]]
car_cords2 = [[0,0]]

x_values = np.arange(0, 641, 40)
y_values = np.arange(0, 481, 40)
# 生成二维网格
grid_x, grid_y = np.meshgrid(x_values, y_values)
grid_points = np.stack((grid_x, grid_y), axis=-1)


# 鼠标回调函数，记录鼠标点击的位置
def get_mouse_click1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测左键按下事件
        print(f"Mouse clicked at position: ({x}, {y})")
        calibration_points_fr1.append([x, y]) # 左上 右上 右下
        print(len(calibration_points_fr1))
def get_mouse_click2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测左键按下事件
        print(f"Mouse clicked at position: ({x}, {y})")
        calibration_points_fr2.append([x, y]) # 左上 右上 右下
        print(len(calibration_points_fr2))

def no_mouse_click(event, x, y, flags, param): 
    pass


def draw_point_on_frame(frame, point, color=(0, 255, 0), radius=5, thickness=-1):
    # 绘制圆点
    cv2.circle(frame, point, radius, color, thickness)

def nearest_neighbor(grid_points, pt):
     # 将输入坐标转为 NumPy 数组
    pt = np.array(pt)
    
    # 计算每个格点与输入坐标之间的欧几里得距离
    distances = np.linalg.norm(grid_points - pt, axis=-1)
    
    # 找到距离最近的点的索引
    nearest_index = np.unravel_index(np.argmin(distances), distances.shape)
    
    # 返回最近的网格点坐标
    nearest_point = tuple(grid_points[nearest_index])
    
    return nearest_point


cv2.namedWindow('frame1')
cv2.setMouseCallback('frame1', get_mouse_click1)
cv2.namedWindow('frame2')
cv2.setMouseCallback('frame2', get_mouse_click2)
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    if len(calibration_points_fr1) == 3:
        cv2.setMouseCallback("frame1", no_mouse_click)
        transformed_frame1 = apply_affine_transform(input_pts=calibration_points_fr1, output_pts=output_points, frame=frame1)
        transformed_frame1 = cv2.resize(transformed_frame1, (640,480), interpolation=cv2.INTER_AREA)
        carx1, cary1 = detection(frame1)
        if carx1 != None and cary1 != None:
            # 标记目标
            (tx1,ty1) = cord_transform((carx1,cary1),calibration_points_fr1,output_points)
            draw_point_on_frame(transformed_frame1, (int(tx1),int(ty1)))
            cv2.putText(transformed_frame1, "car", (int(tx1)+10,int(ty1)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            car_cords1.append([tx1, ty1])
            NN1 = nearest_neighbor(grid_points,(tx1,ty1))
            draw_point_on_frame(transformed_frame1, NN1,(0,0,255))
            # 黑魔法
            ca1 = kalman_filter(car_cords1)
            KNN1 = nearest_neighbor(grid_points,ca1)
            draw_point_on_frame(transformed_frame1, (int(ca1[0]),int(ca1[1])),(255,0,0))
            draw_point_on_frame(transformed_frame1, KNN1,(255,255,255),2)
            cv2.putText(transformed_frame1,'prediction',(int(ca1[0])+10,int(ca1[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('transform1', transformed_frame1)


    if len(calibration_points_fr2) == 3:
        cv2.setMouseCallback("frame2", no_mouse_click)
        transformed_frame2 = apply_affine_transform(input_pts=calibration_points_fr2, output_pts=output_points, frame=frame2)
        transformed_frame2 = cv2.resize(transformed_frame2, (640,480), interpolation=cv2.INTER_AREA)
        carx2, cary2 = detection(frame2)
        if carx2 != None and cary2 != None:
            # 标记目标
            (tx2,ty2) = cord_transform((carx2,cary2),calibration_points_fr2,output_points)
            draw_point_on_frame(transformed_frame2, (int(tx2),int(ty2)))
            cv2.putText(transformed_frame2, "car", (int(tx2)+10,int(ty2)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            car_cords2.append([tx2, ty2])
            NN2 = nearest_neighbor(grid_points,(tx2,ty2))
            draw_point_on_frame(transformed_frame2, NN2,(0,0,255))
            # 黑魔法
            ca2 = kalman_filter(car_cords2)
            KNN2 = nearest_neighbor(grid_points,ca2)
            draw_point_on_frame(transformed_frame2, (int(ca2[0]),int(ca2[1])),(255,0,0))
            draw_point_on_frame(transformed_frame2, KNN2,(255,255,255),2)
            cv2.putText(transformed_frame2,'prediction',(int(ca2[0])+10,int(ca2[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('transform2', transformed_frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()