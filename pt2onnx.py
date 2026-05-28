from ultralytics import YOLO

# 1. 加载模型
model = YOLO('./x-anylabling/best.pt')

# 2. 导出为 ONNX
# dynamic=False 通常对自动标注工具更友好（固定输入尺寸）
model.export(format='onnx', opset=12, simplify=True)