#!/usr/bin/env python3
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

VALIDATE = True

def main():
    # master_model = YOLO("yolo11n.pt")
    # master_model = YOLO("yolo11n-dist.yaml")
    # master_model = YOLO("./runs/dist/train379/weights/best.pt")
    master_model = YOLO("../weights/yolo11n_2026-01-12-01_ep200.pt")

    master_model.export(format="openvino")
    master_model.export(format="ncnn")
    master_model.export(format="onnx")  # last because of potential overwriting
    # master_model.export(format="onnx", imgsz=640, opset=13, dynamic=False)  # for halio hef

    # do all exports in one go and then benchmark
    # benchmark(master_model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device='cpu')

    print("\n\n--- 📥 Export Complete ---\n\n")

    if VALIDATE:
        models_path = [
            "runs/dist/train379/weights/best_openvino_model",
            "runs/dist/train379/weights/best_ncnn_model",
            "runs/dist/train379/weights/best.onnx",
        ]
        for model_path in models_path:
            model = YOLO(model_path, task="dist")

            model.val(data="../datasets/KITTI.yaml", imgsz=640)

            # pred = model.predict(source="../datasets/KITTI2017/images/train/000011.png", device="cpu")
            # print(pred[0].boxes.xyxy)
            # print("🎯🎯", pred[0].distances)

if __name__ == "__main__":
    main()
    