from ultralytics import YOLO

if __name__ == "__main__":
    # 模型名称和路径列表

    models = [
        ("ResNet", r"./runs/train/fire_rgb/exp_backbone_resnet50_epoch500/weights/best.pt"),
        ("EfficientNet", r"./runs/train/fire_rgb/exp_backbone_efficientnet_epoch500/weights/best.pt"),
        ("Swin_Transformer", r"./runs/train/fire_rgb/exp_backbone_swin_transformer_epoch500/weights/best.pt"),
        ("ConvNeXt", r"./runs/train/fire_rgb/exp_backbone_convnext_epoch500/weights/best.pt"),
        ("YOLOv11", r"./runs/train/fire_rgb/exp_yolo11_epoch500/weights/best.pt"),
        ("YOLOv12", r"./runs/train/fire_rgb/exp_yolo12_epoch500/weights/best.pt"),
        ("YOLOv13", r"./runs/train/fire_rgb/exp_yolo13_epoch500/weights/best.pt"),
        ("YOLOv10", r"./runs/train/fire_rgb/exp_yolo10_epoch500/weights/best.pt"),
        ("YOLOv9", r"./runs/train/fire_rgb/exp_yolov9_epoch500/weights/best.pt"),
        ("YOLOv8", r"./runs/train/fire_rgb/exp_yolov8_epoch500/weights/best.pt"),
        ("YOLOv6", r"./runs/train/fire_rgb/exp_yolov6_epoch500/weights/best.pt"),
        ("YOLOv5", r"./runs/train/fire_rgb/exp_yolov5_epoch500/weights/best.pt"),
        ("CSCConvNeXtv2_CBAM_P2", r"./runs/train/fire_rgb/exp_backbone_CSCConvNeXtv2_p2_CBAM_epoch500/weights/best.pt"),
    ]

    # models = [
    #     ("yolo11", r"./runs/train/uav-hongcun-fire-smoke/exp_yolo11n/weights/best.pt"),
    #     ("yolo11_CSCConvNeXtv2_p2_CBAM", r"./runs/train/uav-hongcun-fire-smoke/exp_yolo11_CSCConvNeXtv2_p2_CBAM/weights/best.pt"),
    # ]

    test_path = "E:/paper_fire_rgb"
    # test_path = "E:/fire_rgb/images/val"

    # 为每个模型进行推理
    for model_name, model_path in models:
        print(f"正在推理模型: {model_name}")
        print(f"模型路径: {model_path}")

        # 加载模型
        model = YOLO(model_path)

        # 推理并保存到以模型名字命名的目录
        results = model(test_path, imgsz=640, save=True, project="./runs/detect", name=model_name)

        print(f"模型 {model_name} 推理完成\n")

    print("所有模型推理完成！")
