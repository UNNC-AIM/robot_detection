import sys
import threading
import msvcrt
import numpy as np
import cv2
from ctypes import *
from detect_function import YOLOv5Detector
 
sys.path.append(r"MvCameraControl_class.py")
from MvCameraControl_class import *

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

g_bExit = False

# 加载模型
weights_path = 'models/car.engine'
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=1, ui=True)

def work_thread(cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
        else:
            print("no data[0x%x]" % ret)
        if g_bExit == True:
            break
 

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


calibration_points = []
output_points = [[0, 0], [1440, 0], [1440, 1080]]
car_cords = []

x_values = np.arange(0, 1441, 180)
y_values = np.arange(0, 1081, 180)
# 生成二维网格
grid_x, grid_y = np.meshgrid(x_values, y_values)
grid_points = np.stack((grid_x, grid_y), axis=-1)


# 鼠标回调函数，记录鼠标点击的位置
def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测左键按下事件
        print(f"Mouse clicked at position: ({x}, {y})")
        calibration_points.append([x, y]) # 左上 右上 右下

def no_mouse_click(event, x, y, flags, param): 
    pass

def image_show(image):
    image = cv2.resize(image, (1440, 1080), interpolation=cv2.INTER_AREA)
    cv2.imshow('test', image)
    k = cv2.waitKey(1) & 0xff

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


# 将相机获得的数据转换成opencv支持的格式
def work_thread2(cam=0, pData=0, nDataSize=0):
    # 输出帧的信息
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    print("work_thread_1!\n")
    img_buff = None
    cv2.namedWindow('test')
    cv2.setMouseCallback('test', get_mouse_click)
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
            nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight* 3
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else: #显示及后处理
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                image_show(image=img_buff)  # 显示图像函数
                if len(calibration_points) == 3:
                    cv2.setMouseCallback("test", no_mouse_click)
                    transformed_frame = apply_affine_transform(input_pts=calibration_points, output_pts=output_points, frame=img_buff)
                    transformed_frame = cv2.resize(transformed_frame, (1440, 1080), interpolation=cv2.INTER_AREA)
                    carx, cary = detection(img_buff)
                    if carx != None and cary != None:
                        # 标记目标
                        (tx,ty) = cord_transform((carx,cary),calibration_points,output_points)
                        draw_point_on_frame(transformed_frame, (int(tx),int(ty)))
                        cv2.putText(transformed_frame, "car", (int(tx)+10,int(ty)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        car_cords.append([tx, ty])
                        NN = nearest_neighbor(grid_points,(tx,ty))
                        draw_point_on_frame(transformed_frame, NN,(0,0,255))
                        # 黑魔法
                        ca = kalman_filter(car_cords)
                        KNN = nearest_neighbor(grid_points,ca)
                        draw_point_on_frame(transformed_frame, (int(ca[0]),int(ca[1])),(255,0,0))
                        draw_point_on_frame(transformed_frame, KNN,(255,255,255),2)
                        cv2.putText(transformed_frame,'prediction',(int(ca[0])+10,int(ca[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.imshow('transform', transformed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
 
 
if __name__ == "__main__":
    # 获得设备信息
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
 
    # 枚举设备
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()
 
    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()
 
    print("Find %d devices!" % deviceList.nDeviceNum)
 
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
 
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 选择设备
    nConnectionNum = input("please input the number of the device to connect:")
 
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()
 
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
 
    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
 
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()
 
    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()
 
    # 探测网络最佳包大小(只对GigE相机有效)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
 
    # 设置触发模式为off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
 
    # 从这开始，获取图片数据
    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    # csharp中没有memset函数，用什么代替？
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
 
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    # 关键句，官方没有这个句子，抓取的图片数据是空的，nCurValue是int64
    nPayloadSize = stParam.nCurValue
 
    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    #  关键句，官方没有这个句子，抓取的图片数据是空的，CSharp中怎么实现这句话。
    data_buf = (c_ubyte * nPayloadSize)()
    #  date_buf前面的转化不用，不然报错，因为转了是浮点型
    try:
        hThreadHandle = threading.Thread(target=work_thread2, args=(cam, data_buf, nPayloadSize))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")
 
    print("press a key to stop grabbing.")
    msvcrt.getch()
 
    g_bExit = True
    hThreadHandle.join()
 
    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
 
    # 关闭设备
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
 
    # 销毁句柄
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
 
    del data_buf