import torch
import cv2
import requests
import time
import os
import numpy as np
from ultralytics.utils.plotting import colors


class BirdEyeView:
    """
    Bird's Eye View visualization
    """
    def __init__(self, size=(80, 80), scale=10, names=None):
        """
        Initialize the Bird's Eye View visualizer
        
        Args:
            size (tuple): Size of the BEV image in meters (width, height)
            scale (float): Scale factor (pixels per meter)
        """
        self.scale = scale
        self.size = size
        self.width = int(size[0] * scale)
        self.height = int(size[1] * scale)
        self.names = names # class names
        
        # Create empty BEV image
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Set origin at the bottom center of the image
        self.origin_x = self.width // 2
        self.origin_y = self.height
    
    def reset(self):
        """
        Reset the BEV image
        """
        # Create a dark background
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.bev_image[:, :] = (20, 20, 20)  # Dark gray background
        
        # Draw grid lines
        grid_spacing = int(self.scale * 5)  # every 5 meters
        
        # Draw horizontal grid lines
        for y in range(self.origin_y, 0, -grid_spacing):
            cv2.line(self.bev_image, (0, y), (self.width, y), (50, 50, 50), 1)
        
        # Draw vertical grid lines
        for x in range(0, self.width, grid_spacing):
            cv2.line(self.bev_image, (x, 0), (x, self.height), (50, 50, 50), 1)
        
        # Draw ego vehicle at origin
        size = 10
        cv2.rectangle(self.bev_image,
                      (self.origin_x - size, self.origin_y - size * 2),
                      (self.origin_x + size, self.origin_y),
                      (255, 255, 255), -1) # white
        
        # Draw distance markers
        for dist in range(0, self.size[1], 1):
            display_number = dist % 5 == 0
            y = self.origin_y - int(dist * self.scale)
            
            # Draw tick mark - thicker for whole meters
            thickness = 2 if display_number else 1
            cv2.line(self.bev_image, 
                    (self.origin_x - 5, y), 
                    (self.origin_x + 5, y), 
                    (120, 120, 120), thickness)
            
            # Only show text for whole meters
            if display_number:
                cv2.putText(self.bev_image, f"{int(dist)}", 
                           (self.origin_x + 10, y + 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    def draw_box(self, box_3d, color=None):
        """
        Draw a more realistic representation of an object on the BEV image
        
        Args:
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format (None for automatic color based on class)
        """
        try:            
            obj_id = box_3d['object_id']
            distance = box_3d['distance']
            x1, _, x2, _ = box_3d['bbox']
            img_w, _ = box_3d.get('img_size', (1280, 720))
            
            # Calculate Y position (upward) based on depth
            bev_y = self.origin_y - int(distance * self.scale)
            
            # Calculate X position (rightward) based on horizontal position in image
            center_x_2d = (x1 + x2) / 2
            rel_x = (center_x_2d / img_w) - 0.5
            bev_x = self.origin_x + int(rel_x * self.width * 0.6)

            # Ensure the object stays within the visible area
            # bev_x = max(20, min(bev_x, self.width - 20))
            # bev_y = max(20, min(bev_y, self.origin_y - 10))
            
            # Draw a circle for object
            if color is None:
                color = colors(int(obj_id), True)
            else:
                color = (0, 0, 255)  # Default to red
            radius = int(1.5 * self.scale)
            cv2.circle(self.bev_image, (bev_x, bev_y), radius, color, -1)
            
            # Draw object ID if available
            # if self.names is not None and obj_id is not None:
            #     cv2.putText(self.bev_image, f"{self.names.get(obj_id, 'N/A')}",
            #                (bev_x, bev_y),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw distance line from origin to object
            cv2.line(self.bev_image, 
                    (self.origin_x, self.origin_y),
                    (bev_x, bev_y),
                    (70, 70, 70), 1)
            
        except Exception as e:
            print(f"Error drawing box in BEV: {e}")
    
    def get_image(self):
        """
        Get the BEV image
        
        Returns:
            numpy.ndarray: BEV image
        """
        return self.bev_image
    

def send_telemetry_message(msg):
    try:
        res = requests.post("https://x.com/telegram/sendMessage", json={"message": msg})
        if res.status_code != 200:
            raise Exception(f"Error {res.status_code}")
    except Exception as e:
        print(f"Failed to send telemetry message: {e}")


def detect_objects(model, input_path, target_fps=24, show_conf=False, show_bev=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    print(f"Using device: {device}")

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    if show_bev:
        bev_width = 500
        bev_height = 500
        
        # bev = BirdEyeView(names=model.model.names)
        bev = BirdEyeView()

    # --- VIDEO MODE ---
    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            results = model.predict(frame, device=device)
            annotated_frame = results[0].plot(conf=show_conf)

            infer_time = time.time() - frame_start
            sleep_time = (1.0 / target_fps) - infer_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLO Detection Result", annotated_frame)

            if show_bev:
                res = results[0]

                bev.reset()
                for i in range(len(res.boxes.xyxy)):
                    bev.draw_box({
                        "object_id": res.boxes.cls[i],
                        "bbox": res.boxes.xyxy[i],
                        "distance": res.distances[i],
                        "img_size": res.orig_shape[::-1]
                    })

                bev_image = bev.get_image()
                bev_image = cv2.resize(bev_image, (bev_width, bev_height))

                cv2.imshow("YOLO Bird's Eye View", bev_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames in {time.time() - start_time:.2f}s")

    # --- IMAGE MODE ---
    else:
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Unable to read image.")
            return

        result = model.predict(img, device=device)[0]
        # print(result.distances)
        # print(result.boxes.xywh[0])
        annotated_img = result.plot(conf=show_conf)

        # Show result
        cv2.imshow("YOLO Detection Result", annotated_img)

        if show_bev:
            bev.reset()
            for i in range(len(result.boxes.xyxy)):
                bev.draw_box({
                    "object_id": result.boxes.cls[i],
                    "bbox": result.boxes.xyxy[i],
                    "distance": result.distances[i],
                    "img_size": result.orig_shape[::-1]
                })

            bev_image = bev.get_image()
            bev_image = cv2.resize(bev_image, (bev_width, bev_height))

            cv2.imshow("YOLO Bird's Eye View", bev_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save
        # save_path = "output_detected.jpg"
        # cv2.imwrite(save_path, annotated_img)
        # print(f"Saved detection result to {save_path}")


# arg1 = sys.argv[1] if len(sys.argv) > 1 else None
# arg2 = sys.argv[2] if len(sys.argv) > 2 else None

# if __name__ == "__main__":
#     weights_path = arg1 if arg1 else "./best.pt"
#     # input_path = "datasets/000007.png"
#     # input_path = "datasets/new-york.mp4"
#     input_path = arg2 if arg2 else "../datasets/new-york.mp4"
#     # input_path = arg2 if arg2 else "../datasets/kitti-sequence2.mp4"
    
#     model = YOLO(weights_path, verbose=True)
#     detect_objects(model, input_path)
    