import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    # model = YOLO(model=r'./ultralytics/cfg_yolo11/YOLO11-Backbone/ConvNeXtv2/yolo11s-CSCConvNeXt.yaml')
    # model = YOLO(model=r'runs/train/fire_rgb/exp_yolo13/weights/last.pt')
    model = YOLO(model=r"./ultralytics/cfg/models/djy/yolo11n_50.yaml")
    # model = YOLO(model=r'./ultralytics/cfg/models/11/yolo11s.yaml')
    model.train(
        data=r"./ultralytics/cfg/datasets/djy/fire_rgb.yaml",
        # data=r'./ultralytics/cfg/datasets/djy/RGBT-3M.yaml',
        imgsz=640,
        epochs=2,
        batch=8,
        workers=0,
        device="",
        optimizer="SGD",
        close_mosaic=10,
        resume=True,
        project="runs/train/ceshi/",
        name="exp_yolov11_efficientnet_epoch500",
        single_cls=False,
        cache=False,
        save_period=1,  # 每轮保存一次
    )
