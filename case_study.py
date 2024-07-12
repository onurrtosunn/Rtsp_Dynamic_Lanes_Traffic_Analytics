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
    YOLOV7 model loaded and run RTSP stream.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()
        self.car_status = {}
        self.trackers = {}
        self.stopped_threshold = 10  # Eski frame'e göre durma eşiği, piksel cinsinden
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    def load_model(self):
        """
        Load the YOLO model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torch.load(self.model_name, map_location=device) if isinstance(self.model_name, str) else self.model_name
        if isinstance(self.model, dict):
            self.model = self.model['ema' if self.model.get('ema') else 'model']

        hub_model = Model(self.model.yaml).to(next(self.model.parameters()).device)
        hub_model.load_state_dict(self.model.float().state_dict())
        self.loaded_model = hub_model.autoshape()

    def load_video(self, video_source):
        """ 
        Load video from RTSP stream
        """
        self.cap = cv2.VideoCapture(video_source)

    def yolo_inference_and_drawer(self, output_rtsp_url, video_filename="Traffic.mp4", video_path=os.getcwd()):
        """
        Read frames from a video file and perform YOLO inference on each frame.
        """
        self.detection_results_total = []
        self.video_filename = video_filename
        video_path = join(video_path, video_filename)
        self.frame_count = 0

        if not self.cap.isOpened():
            print("Could not open the video file")
            return

        self.get_video_properties(self.cap)
        video_writer = self.initialize_rtsp_writer(output_rtsp_url)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_detections = self.perform_inference(frame)
            if frame_detections:
                self.detection_results_total.extend(frame_detections)
                self.draw_detections(frame, frame_detections)
            video_writer.write(frame)
            self.frame_count += 1

        self.save_video(self.cap, video_writer)
        return self.detection_results_total

    def get_video_properties(self, cap):
        """
        Get the properties of the video such as fps, width, and height.
        """
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initialize_rtsp_writer(self, output_rtsp_url):
        """
        Initialize the RTSP writer.
        """
        video_writer = cv2.VideoWriter(output_rtsp_url, cv2.VideoWriter_fourcc(*'X264'), self.fps, (self.width, self.height))
        return video_writer

    def save_video(self, cap, video_writer):
        """
        Release the video capture and writer.
        """
        cap.release()
        video_writer.release()

    def perform_inference(self, frame, confidence_threshold=0.50):
        """
        Perform YOLO inference on the given frame.
        """
        detection_results = []
        fg_mask = self.back_sub.apply(frame)
        converted_img = Image.fromarray(frame[:, :, ::-1], 'RGB')
        detection_object = self.loaded_model(converted_img)
        detection_result = detection_object.xyxy[0].cpu().numpy()

        for obj in detection_result:
            x1, y1, x2, y2, conf, class_id = obj
            if conf >= confidence_threshold:
                class_name = self.model.names[int(class_id)]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                if fg_mask[y1:y2, x1:x2].mean() > 35:  # Adjust the threshold value as needed
                    detection_results.append([class_name, x1, y1, x2, y2, conf, True])  # Mark as moving
                else:
                    detection_results.append([class_name, x1, y1, x2, y2, conf, False])  # Mark as not moving

        return detection_results

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        """
        for detection in detections:
            class_name, x1, y1, x2, y2, confidence, is_moving = detection
            color = (0, 255, 0) if is_moving else (0, 0, 255)
            self.draw_rectangle_to_image(frame, class_name, (x1, y1, x2, y2), confidence, color=color)

    def draw_rectangle_to_image(self, img, class_name, bbox, confidence, line_thickness=3, color=None):
        """
        Draws boxes around objects detected in the image based on their locations.
        """
        line_thickness = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        label = f'{class_name} [{confidence:.2f}]'
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_thickness)
        background_color = (0, 0, 0)
        text_color = (255, 255, 255)
        bbox_color = color or [random.randint(0, 255) for _ in range(3)]

        c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        text_position = (c1[0], c1[1] - 2)

        cv2.rectangle(img, (c1[0], c1[1] - text_size[1] - 3), (c1[0] + text_size[0], c1[1] - 2), background_color, -1)
        cv2.rectangle(img, c1, c2, bbox_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

# Kullanım örneği
model_name = 'yolov7.pt'
video_source = 'rtsp://localhost:9090/mystream'
output_rtsp_url = 'rtsp://localhost:8554/mystream'

yolo_detector = YOLODetection(model_name)
yolo_detector.load_video(video_source)
yolo_detector.yolo_inference_and_drawer(output_rtsp_url)
