import os
import cv2
import torch
import numpy as np
from PIL import Image
from models.yolo import Model
from sort import Sort
import subprocess as sp
from collections import defaultdict

class YOLODetection:
    def __init__(self, model_name):
        self.model_name = model_name
        self.stopped_threshold = 3
        self.background_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.stopped_objects = defaultdict(int)  
        self.load_model()
        self.initialize_background_subtraction()
        self.initialize_sort_tracker()

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_name, map_location=device) if isinstance(self.model_name, str) else self.model_name
        if isinstance(self.model, dict):
            self.model = self.model.get('ema' if self.model.get('ema') else 'model')
        hub_model = Model(self.model.yaml).to(device)
        hub_model.load_state_dict(self.model.float().state_dict())
        self.loaded_model = hub_model.autoshape()

    def initialize_background_subtraction(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    def initialize_sort_tracker(self):
        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    def load_video(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Could not open the video source")
        self.get_video_properties()

    def get_video_properties(self):
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def perform_inference(self, frame):

        converted_img = Image.fromarray(frame[:, :, ::-1], 'RGB')
        detection_object = self.loaded_model(converted_img)
        detection_result = detection_object.xyxy[0].cpu().detach().numpy()
        fg_mask = self.back_sub.apply(frame)
        return self.analyzer(fg_mask, detection_result)

    def analyzer(self, fg_mask, detection_result, confidence_threshold=0.40):

        detections_to_sort = []
        for obj in detection_result:

            x1, y1, x2, y2, confidence, class_id = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if confidence >= confidence_threshold:
                detections_to_sort.append([x1, y1, x2, y2, confidence, class_id])

                if fg_mask[y1:y2, x1:x2].mean() > 0.5:
                    obj_id = (x1, y1, x2, y2)
                    self.stopped_objects[obj_id] += 1
                else:
                    obj_id = (x1, y1, x2, y2)
                    if obj_id in self.stopped_objects:
                        del self.stopped_objects[obj_id]

        detections_to_sort = np.array(detections_to_sort)
        return detections_to_sort

    def draw_detections(self, frame, detections):
        bbox_xyxy = detections[:, :4]
        obj_ids = detections[:, -1].astype(int)
        class_ids = detections[:, 4].astype(int)

        self.draw_rectangle_to_image(frame, bbox_xyxy, obj_ids, class_ids)

    def draw_rectangle_to_image(self, img, bbox, obj_ids, class_ids, line_thickness=3):

        for bbox, obj_id, class_id in zip(bbox, obj_ids, class_ids):
            x1, y1, x2, y2 = [int(i) for i in bbox]

            label = f"ID: {obj_id} - Class: {self.model.names[class_id]}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_thickness)
            text_position = (x1, y1 - 2)

            obj_coords = (x1, y1, x2, y2)
            if self.stopped_objects.get(obj_coords, 0) >= self.stopped_threshold:
                bbox_color = (0, 0, 255)  
            else:
                bbox_color = (0, 255, 0) 

            cv2.rectangle(img, (x1, y1 - text_size[1] - 3), (x1 + text_size[0], y1 - 2), self.background_color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.text_color, 1)

    def track_objects_with_sort(self, detections):
        
        return self.sort_tracker.update(detections)

    def stream_video_with_detections_and_sort(self, rtsp_url):

        self.start_ffplay(rtsp_url)
        self.start_ffmpeg(rtsp_url)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.perform_inference(frame)
            tracked_detections = self.track_objects_with_sort(detections)
            if len(tracked_detections) > 0:
                self.draw_detections(frame, tracked_detections)

            self.write_frame_to_ffmpeg(frame)

        self.release_resources()

    def start_ffplay(self, rtsp_url):
        self.ffplay_process = sp.Popen(['ffplay', '-rtsp_flags', 'listen', rtsp_url])

    def start_ffmpeg(self, rtsp_url):
        command = [
            'ffmpeg', '-re', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}", '-r', str(self.fps), '-i', '-', '-c:v', 'libx264',
            '-preset', 'ultrafast', '-f', 'rtsp', '-rtsp_transport', 'tcp', '-muxdelay', '0.1',
            '-bsf:v', 'dump_extra', rtsp_url
        ]
        self.ffmpeg_process = sp.Popen(command, stdin=sp.PIPE)

    def write_frame_to_ffmpeg(self, frame):
        self.ffmpeg_process.stdin.write(frame.tobytes())

    def release_resources(self):
        self.cap.release()
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
        self.ffplay_process.terminate()

if __name__ == "__main__":
    model_name = 'yolov7.pt'
    video_source = 'Traffic.mp4'
    output_rtsp_url = 'rtsp://localhost:8080/output'

    yolo_detector = YOLODetection(model_name)
    yolo_detector.load_video(video_source)
    yolo_detector.stream_video_with_detections_and_sort(output_rtsp_url)
