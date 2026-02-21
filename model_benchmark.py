#!/usr/bin/env python3
import time
import tqdm
import random
import logging
import os
import psutil
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.benchmarks import benchmark


# model's accuracy (precision, mAP, MAE, etc.) on the given dataset
def measure_model_perf(dataset=None):
    if dataset is None:
        raise ValueError("Dataset path must be provided")

    model = YOLO(MODEL, task=TASK)
    
    results = model.val(
        data=dataset,
        imgsz=640,
        plots=False,
        device=DEVICE,
        half=False,
        int8=False,
        verbose=False,
        conf=0.001,  # all the pre-set benchmark mAP values are based on conf=0.001
    )

    print("\n===== Validation Results =====")
    for key in results.keys:
        print(f"{key:25}: {results.results_dict[key]:.4f}")
    print("\n")


# computer's performance (FPS, latency, etc.) on the given model and set of images
def measure_computational_perf(cam_id=0, images_path=None, sample_size=1000, duration=30):
    proc = psutil.Process(os.getpid())  # get current process for CPU and memory measurement

    model = YOLO(MODEL, task=TASK)

    is_using_camera = images_path is None
    if is_using_camera:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
    else:
        # initialize validation dataset
        dataset = YOLODataset(
            img_path=images_path,
            augment=False,
            data={'nc': 7}
        )
        sample_data = random.sample(range(dataset.ni), sample_size)

    inf_lat_samples = []  # inference latencies in ms
    e2e_lat_samples = []  # end2end frame latencies in ms
    cpu_samples = []
    mem_samples = []

    # turn off inference output during benchmarking
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # warm-up the model
    if is_using_camera:
        ok, warmup_frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam")
    else:
        warmup_frame = dataset.get_image_and_label(sample_data[0])["img"]
    for _ in range(10):
        model.predict(warmup_frame, device=DEVICE, verbose=False)

    proc.cpu_percent(interval=None)  # initialize CPU measurement
    start_time = time.monotonic()
    frame_count = 0
    last_cpu_usage = 0.0  # just for live update display
    
    def process_frame(frame):
        nonlocal frame_count, last_cpu_usage

        result = model.predict(frame, device=DEVICE, verbose=False)[0]
        
        pre_latency = result.speed['preprocess']  # preprocess latency in ms
        inf_latency = result.speed['inference']  # inference latency in ms
        post_latency = result.speed['postprocess']  # postprocess latency in ms
        inf_lat_samples.append(inf_latency)
        e2e_lat_samples.append(pre_latency + inf_latency + post_latency)

        # sample CPU and memory every 10 frames to reduce overhead
        if frame_count % 10 == 0:
            cpu_usage = proc.cpu_percent(interval=None)
            last_cpu_usage = cpu_usage
            cpu_samples.append(cpu_usage)
            mem_samples.append(proc.memory_info().rss / 1024**2)

        frame_count += 1

        # live update progress bar
        stats = {
            "lat(ms)": f"{inf_latency:.1f}",
            "CPU(%)": f"{last_cpu_usage:.1f}"
        }
        if is_using_camera:
            stats["frame"] = frame_count
        pbar.set_postfix(stats)

    if is_using_camera:
        pbar = tqdm.tqdm(total=duration)
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            pbar.n = int(elapsed)
            if elapsed >= duration:  # time's up
                break

            ok, frame = cap.read()
            if not ok:
                continue

            process_frame(frame)
            # manually update time-based progress bar
            pbar.refresh()

        cap.release()
        pbar.close()

    else:
        pbar = tqdm.tqdm(sample_data, unit="frame")
        for idx in pbar:
            label = dataset.get_image_and_label(idx)
            process_frame(label["img"])

    total_time = time.monotonic() - start_time

    logging.getLogger("ultralytics").setLevel(logging.INFO)

    print("\n===== Computation Results =====")
    print(f"Total time          : {total_time:.2f} s")
    print(f"Frames processed    : {frame_count}")
    print(f"Average FPS         : {frame_count / total_time:.2f}")
    print(f"E2E Mean latency    : {np.mean(e2e_lat_samples):.2f} ms")
    print(f"E2E P95 latency     : {np.percentile(e2e_lat_samples, 95):.2f} ms")
    print(f"Infer Mean latency  : {np.mean(inf_lat_samples):.2f} ms")
    print(f"Infer P95 latency   : {np.percentile(inf_lat_samples, 95):.2f} ms")
    print(f"Mean CPU usage      : {np.mean(cpu_samples):.2f} %")
    print(f"Peak CPU usage      : {np.max(cpu_samples):.2f} %")
    print(f"Mean Memory usage   : {np.mean(mem_samples):.2f} MB")
    print(f"Peak Memory usage   : {np.max(mem_samples):.2f} MB")
    print("\n")


if __name__ == "__main__":
    # MODEL = "./yolo11n.pt"
    # MODEL = "./yolo11n.onnx"
    # MODEL = "./yolo11n_openvino_model"
    # MODEL = "./yolo11n_ncnn_model"
    MODEL = "../weights/yolo11n-dist_2026-01-12-02_ep200.pt"
    # MODEL = "./runs/dist/train392/weights/best.pt"
    TASK = "dist"
    DEVICE = "cpu"

    dataset_path = "/home/riset/Documents/zherk/datasets"

    print("===== Starting Benchmark =====")
    print(f"Model  : {MODEL}")
    print(f"Task   : {TASK}")
    print(f"Device : {DEVICE}")
    print("\n")

    measure_model_perf(dataset=dataset_path + "/KITTI.yaml")
    measure_computational_perf(cam_id=0, duration=30)
    measure_computational_perf(images_path=dataset_path + "/KITTI2017/images/val", sample_size=1000)
    
    # benchmark(model="./yolo11n.pt", data="coco8.yaml", imgsz=640, device=DEVICE)
    # benchmark(model="./yolo11n-dist.pt", data="KITTI.yaml", imgsz=640, device=DEVICE)
