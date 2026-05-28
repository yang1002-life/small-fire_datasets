import time

import pandas as pd
import torch
import torch.nn as nn
from thop import profile

from ultralytics import YOLO


# =====================================================
# 工具函数：清理 THOP 残留属性
# =====================================================
def clean_thop_attrs(model):
    """清理 thop 注册的 total_ops / total_params 防止重复 profile 或 hook 冲突.
    """
    for m in model.modules():
        if hasattr(m, "total_ops"):
            delattr(m, "total_ops")
        if hasattr(m, "total_params"):
            delattr(m, "total_params")


# =====================================================
# 自定义 FLOPs hook（忽略激活层 FLOPs）
# =====================================================
def ignore_flops_counter_hook(m, x, y):
    m.total_ops = torch.zeros(1)
    m.total_params = torch.zeros(1)


custom_ops = {
    nn.ReLU: ignore_flops_counter_hook,
    nn.SiLU: ignore_flops_counter_hook,
    nn.LeakyReLU: ignore_flops_counter_hook,
}

# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":
    # -----------------------------
    # 数据集配置
    # -----------------------------
    # data_yaml = r'./ultralytics/cfg/datasets/djy/VOC2007_small_object_optimize.yaml'
    # data_yaml = r'./ultralytics/cfg/datasets/djy/fire_rgb.yaml'
    data_yaml = r"./ultralytics/cfg/datasets/djy/uav-hongcun-fire-smoke.yaml"

    # -----------------------------
    # 模型列表
    # -----------------------------
    model_info_list = [
        # ("ResNet", r"./runs/train/fire_rgb/exp_backbone_resnet50_epoch500/weights/best.pt"),
        # ("EfficientNet", r"./runs/train/fire_rgb/exp_backbone_efficientnet_epoch500/weights/best.pt"),
        # ("Swin_Transformer", r"./runs/train/fire_rgb/exp_backbone_swin_transformer_epoch500/weights/best.pt"),
        # ("ConvNeXt", r"./runs/train/fire_rgb/exp_backbone_convnext_epoch500/weights/best.pt"),
        # ("YOLOv11", r"./runs/train/fire_rgb/exp_yolo11_epoch500/weights/best.pt"),
        # ("YOLOv12", r"./runs/train/fire_rgb/exp_yolo12_epoch500/weights/best.pt"),
        # ("YOLOv13", r"./runs/train/fire_rgb/exp_yolo13_epoch500/weights/best.pt"),
        # ("YOLOv10", r"./runs/train/fire_rgb/exp_yolo10_epoch500/weights/best.pt"),
        # ("YOLOv9", r"./runs/train/fire_rgb/exp_yolov9_epoch500/weights/best.pt"),
        # ("YOLOv8", r"./runs/train/fire_rgb/exp_yolov8_epoch500/weights/best.pt"),
        # ("YOLOv6", r"./runs/train/fire_rgb/exp_yolov6_epoch500/weights/best.pt"),
        # ("YOLOv5", r"./runs/train/fire_rgb/exp_yolov5_epoch500/weights/best.pt"),
        (
            "CSCConvNeXtv2_CBAM_P2",
            r"./runs/train/uav-hongcun-fire-smoke/exp_yolo11_CSCConvNeXtv2_p2_CBAM/weights/best.pt",
        ),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # =================================================
    # 循环评估模型
    # =================================================
    for model_name, weight_path in model_info_list:
        print(f"\n🚀 正在评估模型: {model_name}")

        # =================================================
        # 1️⃣ FLOPs / Params（专用模型）
        # =================================================
        model_flops = YOLO(weight_path)
        model_f_flops = model_flops.model.to(device).eval()

        dummy_input = torch.randn(1, 3, 640, 640).to(device)

        # 清理 thop 残留
        clean_thop_attrs(model_f_flops)

        with torch.no_grad():
            flops, params = profile(model_f_flops, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)

        gflops = flops / 1e9
        params_m = params / 1e6

        # ❗ FLOPs 模型到此结束
        del model_flops, model_f_flops
        torch.cuda.empty_cache()

        # =================================================
        # 2️⃣ FPS + Validation（干净模型）
        # =================================================
        model = YOLO(weight_path)
        model_f = model.model.to(device).eval()

        # ---------------- FPS ----------------
        with torch.no_grad():
            for _ in range(10):  # warmup
                model_f(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                model_f(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        fps = 100 / (end - start)

        # ---------------- Validation ----------------
        metrics = model.val(data=data_yaml, save_json=False, verbose=False)

        precision = metrics.box.mp
        recall = metrics.box.mr
        f1_score = (2 * precision * recall) / (precision + recall + 1e-6)

        # =================================================
        # 3️⃣ 保存结果
        # =================================================
        results.append(
            {
                "Model": model_name,
                "FPS": round(fps, 3),
                "GFLOPs": round(gflops, 3),
                "Params(M)": round(params_m, 2),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1_score, 4),
                "mAP@0.5": round(metrics.box.map50, 4),
                "mAP@0.75": round(metrics.box.map75, 4),
                "mAP@0.5:0.95": round(metrics.box.map, 4),
            }
        )

        print(f"✅ {model_name} | FPS: {fps:.2f} | GFLOPs: {gflops:.2f} | Params: {params_m:.2f}M")

    # =================================================
    # 4️⃣ 导出 Excel
    # =================================================
    df = pd.DataFrame(results)
    output_path = "./model_eval_results.xlsx"
    df.to_excel(output_path, index=False)

    print(f"\n📊 所有模型评估完成，结果已保存到 {output_path}")
    print(df)
