#!/usr/bin/env python3
from ultralytics import YOLO
from torchinfo import summary
from test_utils import detect_objects, send_telemetry_message
from ultralytics.utils.benchmarks import benchmark

dataset_path = 'D:\\UGM\\tugas akhir\\3. skripsi\\code\\datasets'
# dataset_path = '/home/riset/Documents/zherk/datasets'

def on_epoch_end(trainer):
    ep = trainer.epoch + 1
    eps = trainer.epochs
    # if ep == 1 or ep % 50 == 0 or ep == eps:
    if ep % 50 == 0 or ep == eps:
        send_telemetry_message(f"Epoch {ep} of {eps} done")
        
def main():
    # model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n-dist.yaml")
    # model = YOLO("yolo11n-obb.yaml")
    # model = YOLO("yolo11n-pose.yaml")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11n-obb.pt")

    # model.info(verbose=True, detailed=True)
    # summary(model.model, input_size=(1, 3, 640, 640))
    
    # model.add_callback("on_train_epoch_end", on_epoch_end)

    # model.load(weights="../weights/yolo11n.pt")
    # model.load(weights="../weights/yolo11n_2026-01-12-01_ep200.pt")
    # model.load(weights="../weights/yolo11n-dist_2026-01-12-02_ep200.pt")
    # model.load(weights="./runs/dist/train435/weights/best.pt")
    # model.load(weights="./best.pt")

    # phase 1 (yolo11n)
    # results = model.train(data=dataset_path + "/KITTI.yaml", epochs=200, imgsz=640, batch=32, workers=12,
    #                       optimizer="AdamW", lr0=0.0009, momentum=0.9, warmup_bias_lr=0.0)
    # phase 2 (yolo11n-dist)
    # results = model.train(data=dataset_path + "/KITTI.yaml", epochs=200, imgsz=640, batch=32, workers=12, freeze=9,
    #                       optimizer="AdamW", lr0=0.0009, momentum=0.9, warmup_bias_lr=0.0)
    # results = model.train(data=dataset_path + "/KITTI.yaml", epochs=5, imgsz=640)
    # results = model.train(data=dataset_path + "/coco8-dist.yaml", epochs=5, imgsz=640)
    # results = model.train(data=dataset_path + "/coco8.yaml", epochs=5, imgsz=640)
    # results = model.train(data=dataset_path + "/coco8-pose.yaml", epochs=5, imgsz=640)
    # print(results)

    # metrics = model.val(data=dataset_path + "/KITTI.yaml", imgsz=640, batch=32, use_euclidean=False)
    # metrics = model.val(data=dataset_path + "/coco8-dist.yaml", imgsz=640, batch=16)
    # metrics = model.val(data=dataset_path + "/coco8.yaml", imgsz=640, batch=16)
    # print(metrics)
    
    # detect_objects(model, "../datasets/street.jpg")
    # detect_objects(model, "../datasets/005992.png")
    # detect_objects(model, "../datasets/000072.png")
    # detect_objects(model, "../datasets/new-york.mp4")
    # detect_objects(model, "../datasets/kitti-track-video/0014.mp4", target_fps=10, show_bev=True)

    # model.export(format="onnx", imgsz=640, opset=13, dynamic=False)
    # model.export(format="ncnn")

    # benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device='cpu')

if __name__ == "__main__":
    main()
    