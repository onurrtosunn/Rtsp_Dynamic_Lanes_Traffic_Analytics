import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from models.yolo import Model
from os.path import join, dirname, basename, splitext


class YOLODetection:
    """
    YOLOV7 model loaded and run on the specified image.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """
        Load the YOLO model.
        """
        self.model = torch.load(self.model_name, map_location=torch.device(0)) if isinstance(self.model_name, str) else self.model_name
        if isinstance(self.model, dict):
            self.model = self.model['ema' if self.model.get('ema') else 'model']
        hub_model = Model(self.model.yaml).to(next(self.model.parameters()).device)
        hub_model.load_state_dict(self.model.float().state_dict())
        self.loaded_model = hub_model.autoshape()

    def yolo_inference_and_drawer(self, cap, video_filename, video_path):
        """
        Read frames from a video file and perform YOLO inference on each frame.
        """
        self.detection_results_total = []
        self.video_filename = video_filename
        video_path = join(video_path, video_filename)

        if not cap.isOpened():
            print("Could not open the video file")
            return
        
        self.get_video_properties(cap)
        video_writer, output_path = self.initialize_video_writer(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_detections = self.perform_inference(frame)
            if frame_detections:
                self.detection_results_total.extend(frame_detections)
                self.draw_detections(frame, frame_detections)
            video_writer.write(frame)

        self.save_video(cap, video_writer, video_path, output_path)
        return self.detection_results_total

    def get_video_properties(self, cap):
        """
        Get the properties of the video such as fps, width, and height.
        """
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initialize_video_writer(self, video_path):
        """
        Initialize the video writer.
        """
        output_path = join(dirname(video_path), f"{splitext(basename(video_path))[0]}_output.mp4")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        return video_writer, output_path

    def save_video(self, cap, video_writer, video_path, output_path):
        """
        Release the video capture and writer, and rename the output file.
        """
        cap.release()
        video_writer.release()
        new_video_path = video_path.replace("_output", "")
        os.remove(video_path)
        os.rename(output_path, new_video_path)

    def perform_inference(self, frame, confidence_threshold=0.50):
        """
        Perform YOLO inference on the given frame.
        """
        detection_results = []
        converted_img = Image.fromarray(frame[:, :, ::-1], 'RGB')
        detection_object = self.loaded_model(converted_img)
        detection_result = detection_object.xyxy[0].cpu().numpy()
        
        for obj in detection_result:
            x1, y1, x2, y2, conf, class_id = obj
            if conf >= confidence_threshold:
                class_name = self.model.names[int(class_id)]
                detection_results.append([class_name, x1, y1, x2, y2, conf])
        
        return detection_results

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        """
        for class_name, x1, y1, x2, y2, confidence in detections:
            self.draw_rectangle_to_image(frame, class_name, (x1, y1, x2, y2), confidence)

    def draw_rectangle_to_image(self, img, class_name, bbox, confidence, line_thickness=3, color=None):
        """
        Draws boxes around objects detected in the image based on their locations.
        """
        line_thickness = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        label = f'{class_name} [{confidence:.2f}]'
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_thickness)
        background_color = (0, 0, 0)
        text_color = (255, 255, 255)
        bbox_color = (0, 0, 255)
        color = color or [random.randint(0, 255) for _ in range(3)]

        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        text_position = (c1[0], c1[1] - 2)

        cv2.rectangle(img, (c1[0], c1[1] - text_size[1] - 3), (c1[0] + text_size[0], c1[1] - 2), background_color, -1)
        cv2.rectangle(img, c1, c2, bbox_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
