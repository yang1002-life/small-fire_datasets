import netron
import torch
from PIL import Image
import onnx
import sys
import os
import numpy as np
from pathlib import Path
from typing import Union
import cv2
from ultralytics import YOLO


def train():
    # 加载模型配置文件，这里使用v8的m模型结构
    model = YOLO('yolov8m.yaml')

    # 做预训练
    # model = YOLO('yolov8x.pt')
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 训练模型
    model.train(data="coco.yaml", epochs=100, imgsz=640)


def onnx():
    # 使用onnx导出文件
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('YOLOv8/runs/detect/train1/weights/best.pt')  # load a custom trained
    # Export the model
    model.export(format='onnx')


def test_img():
    # 训练好的模型权重路径
    model = YOLO("YOLOv8/runs/detect/train1/weights/best.pt")
    # 测试图片的路径
    img = cv2.imread("YOLOv8/7.jpg")
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 设置保存图片的路径
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    if os.path.exists(cur_path):
        cv2.imwrite(cur_path + os.sep + "out.jpg", ann)
    else:
        os.mkdir(cur_path)
        cv2.imwrite(cur_path + os.sep + "out.jpg", ann)


def predict():
    from ultralytics import YOLO

    # Load a model
    # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
    model = YOLO('YOLOv8/runs/detect/your/weights/best.pt')  # 加载自定义的模型权重作评估

    # 评估
    metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # 包含每个类别的map50-95列表


def test_video():
    model = YOLO("./runs/train/fire_rgb/exp_yolo11_epoch500/weights/best.pt")
    # 测试视频存放目录
    pa = "./video/zjgw.mp4"
    cap = cv2.VideoCapture(pa)
    # 调用设备自身摄像头
    # cap = cv2.VideoCapture(0) # -1
    # 设置视频尺寸
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),)
    # 第一个参数是将检测视频存储的路径
    out = cv2.VideoWriter('save.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res = model(frame)
            ann = res[0].plot()
            cv2.imshow("yolo", ann)
            out.write(ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


def tracker(): #目标跟踪
    pa = "/home/you/Downloads/l.mp4"
    cap = cv2.VideoCapture(pa)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),)
    model = YOLO("YOLOv8/runs/detect/train1/weights/best.pt")
    flag = 0
    out = cv2.VideoWriter('save.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)
    while True:
        if flag < 1:
            flag += 1
            continue
        else:
            flag += 1
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # result.boxes.id.cpu().numpy().astype(int)
            try:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Id {id}",
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                print(e)
            cv2.imshow("frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# train()
test_video()
# test_img()
# predict()
# tracker()
# onnx()

# 下面是使用netron导出模型结构
# netron.start("YOLOv8/runs/detect/train1/weights/best.onnx")
