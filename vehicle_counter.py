import sys
import cv2
import time
import torch
import signal
import requests
from models.yolo import Model
from yolo_detector import YOLODetection

class VideoURLFinder:

    def __init__(self, video_api_url, configs_url):
        self.video_api_url = video_api_url
        self.configs_url = configs_url

    def get_video_url(self):
        response = requests.get(self.video_api_url)
        if response.status_code == 200:
            return response.json().get('video_url')
        else:
            return None
    
    def get_configs(self):
        response = requests.get(self.configs_url)
        if response.status_code == 200:
            data = response.json()
            return data.get('config') 
        else:
            return None

class VideoProcessor:

    def __init__(self, video_source_url_finder, model_name):

        self.video_source_url_finder = video_source_url_finder
        self.current_video_url = None
        self.loop_count = 0
        self.model_name = model_name
        self.load_model()
        self.yolo_detector = YOLODetection()
    
    def load_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_name, map_location=device) if isinstance(self.model_name, str) else self.model_name
        if isinstance(self.model, dict):
            self.model = self.model.get('ema' if self.model.get('ema') else 'model')
        hub_model = Model(self.model.yaml).to(device)
        hub_model.load_state_dict(self.model.float().state_dict())
        self.loaded_model = hub_model.autoshape()
        
    def get_video_properties(self):

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def crop_lines(self, configs):
        print(configs)


    def main(self):

        while True:
            new_video_url = self.video_source_url_finder.get_video_url()
            configs = self.video_source_url_finder.get_configs()

            if not new_video_url:
                print("Failed to retrieve video URL!")
                break

            if new_video_url != self.current_video_url:
                print(f"Video URL updated: {new_video_url}")
                self.current_video_url = new_video_url
                self.loop_count = 0

                self.cap = cv2.VideoCapture(self.current_video_url)
                if not self.cap.isOpened():
                    print("Failed to open video!")
                    break

            ret, frame = self.cap.read()
            if not ret:
                print("Video stream ended or failed to read.")
                continue
            self.get_video_properties()
            self.loop_count += 1
            end_point_1_value = configs['end_point_1']
            print(configs)
            print("*********", end_point_1_value, "**********")

            self.yolo_detector.stream_video_with_detections_and_tracker(frame, self.loaded_model, self.model)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                break

if __name__ == "__main__":
    api_url = 'http://127.0.0.1:5001/get_video_url'
    configs_url = 'http://127.0.0.1:5001/get_config'
    model_name = "yolov7.pt"
    
    video_source_url_finder = VideoURLFinder(api_url, configs_url)
    processor = VideoProcessor(video_source_url_finder, model_name)
    processor.main()
