import sys
import cv2
import time
import threading
import torch
import requests
import numpy as np
from models.yolo import Model
from drawer import Drawer
from yolo_detector import YOLODetection
class VideoURLFinder:
    def __init__(self, video_api_url, configs_url, data_url, false_object_url, stopped_object_url):
        self.video_api_url = video_api_url
        self.configs_url = configs_url
        self.data_url = data_url
        self.false_object_url = false_object_url
        self.stopped_object_url = stopped_object_url

    def fetch_json_data(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
        return None

    def get_video_url(self):
        return self.fetch_json_data(self.video_api_url).get('video_url')

    def get_configs(self):
        return self.fetch_json_data(self.configs_url).get('config')

    def get_lane_count_data(self):
        return self.fetch_json_data(self.data_url).get('data')

    def get_stopped_data(self):
        return self.fetch_json_data(self.stopped_object_url).get('stopped')

    def get_wrong_direction(self):
        return self.fetch_json_data(self.false_object_url).get('false')

class ZoneManager:

    def lanes_info(self, lanes):

        self.lanes = lanes

    def draw_line(self, frame):

        for lane in self.lanes:
            cv2.line(frame, lane["line_1_start"], lane["line_1_end"], (0, 0, 255), 5)
            cv2.line(frame, lane["line_2_start"], lane["line_2_end"], (0, 255, 0), 5)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    def draw_center_point(self, frame, center_x, center_y):

        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

    def zone_counter(self, frame, bboxes, object_ids):

        for bbox, object_id in zip(bboxes, object_ids):
            center_x, center_y = self.find_object_center(bbox)
            self.draw_center_point(frame, center_x, center_y)

            for i, lane in enumerate(self.lanes):
                if self.is_object_on_line(lane["line_1_start"], lane["line_1_end"], center_x, center_y):
                    if object_id not in lane["line_1_counter"]:
                        lane["line_1_counter"].append(object_id)
                        
                        print(f"Lane {i+1} - Object ID {object_id} has entered the 1st zone.")
                        print(f"Lane {i+1} Line 1 counter: {lane['line_1_counter']}")

                if self.is_object_on_line(lane["line_2_start"], lane["line_2_end"], center_x, center_y):
                    if object_id not in lane["line_2_counter"]:
                        lane["line_2_counter"].append(object_id)
                        print(f"Lane {i+1} - Object ID {object_id} has entered the 2nd zone.")
                        print(f"Lane {i+1} Line 2 counter: {lane['line_2_counter']}")

                        if object_id not in lane["line_1_counter"]:
                            print(f"Lane {i+1} - Object ID {object_id} entered line 2 before line 1: Wrong direction.")
                            self.send_wrong_direction_detection(object_id, i+1)
    
    def send_wrong_direction_detection(self, object_id, lane_index):

        try:
            text = f"Object ID {object_id} - Lane {lane_index + 1}"
            response = requests.put("http://127.0.0.1:5001/send_wrong_direction_object", json=text)

            if response.status_code == 200:
                print(text)
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending data: {e}")
     
    def send_lane_counts(self):        

        try:
            lane_counts = self.get_lane_counts()
            response = requests.put("http://127.0.0.1:5001/send_lane_counts", json=lane_counts)

            if response.status_code == 200:
                print(lane_counts)
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending data: {e}")

    def get_lane_counts(self):

        lane_counts = {
            f"lane_{i+1}": len(lane["line_2_counter"]) for i, lane in enumerate(self.lanes)
        }
        return lane_counts

    def is_object_on_line(self, line_start, line_end, center_x, center_y):
        
        if min(line_start[0], line_end[0]) <= center_x <= max(line_start[0], line_end[0]):
            if line_start[1] - 20 <= center_y <= line_start[1] + 20:
                return True
        return False

    def find_object_center(self, bbox):
       
        x1, y1, x2, y2 = [int(i) for i in bbox]
        w, h = x2 - x1, y2 - y1
        center_x, center_y = x1 + w // 2, y1 + h // 2
        return center_x, center_y

class VideoProcessor:

    def __init__(self, video_source_url_finder, zone_manager, model_name):

        self.video_source_url_finder = video_source_url_finder
        self.zone_manager = zone_manager
        self.current_video_url = None
        self.model_name = model_name
        self.load_model()
        self.yolo_detector = YOLODetection()
        self.drawer = Drawer()
        self.formatted_data = []
        self.default_lanes = [
            {
                "line_1_start": (240, 466),  
                "line_1_end": (344, 466),    
                "line_2_start": (240, 446),  
                "line_2_end": (348, 446),    
                "line_1_counter": [],     
                "line_2_counter": []       
            },
            {
                "line_1_start": (365, 466),
                "line_1_end": (460, 466),
                "line_2_start": (365, 446),
                "line_2_end": (460, 446),
                "line_1_counter": [],
                "line_2_counter": []
            },
            {
                "line_1_start": (480, 466),
                "line_1_end": (580, 466),
                "line_2_start": (485, 446),
                "line_2_end": (580, 446),
                "line_1_counter": [],
                "line_2_counter": []
            },
            {
                "line_1_start": (710, 446),
                "line_1_end": (810, 446),
                "line_2_start": (715, 466),
                "line_2_end": (810, 466),
                "line_1_counter": [],
                "line_2_counter": []
            },
            {
                "line_1_start": (830, 446),
                "line_1_end": (930, 446),
                "line_2_start": (835, 466),
                "line_2_end": (930, 466),
                "line_1_counter": [],
                "line_2_counter": []
            },
            {
                "line_1_start": (950, 446),
                "line_1_end": (1050, 446),
                "line_2_start": (955, 466),
                "line_2_end": (1050, 466),
                "line_1_counter": [],
                "line_2_counter": []
            }]
        
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
    
    def is_lane_info_updated(self):

        current_lanes_info = self.video_source_url_finder.get_configs()
        self.formatted_lanes_info = []

        for entry in current_lanes_info:
            formatted_entry = {
                'line_1_start': tuple(entry['line_1_start']),
                'line_1_end': tuple(entry['line_1_end']),
                'line_2_start': tuple(entry['line_2_start']),
                'line_2_end': tuple(entry['line_2_end']),
                'line_1_counter': entry['line_1_counter'],
                'line_2_counter': entry['line_2_counter']
            }
            self.formatted_lanes_info.append(formatted_entry)

        if self.formatted_lanes_info == self.default_lanes:
            return False
        else:
            self.default_lanes = self.formatted_lanes_info
            return True

    def main(self):

        lanes = self.video_source_url_finder.get_configs()
        while True:
        
            new_video_url = self.video_source_url_finder.get_video_url()
            self.video_source_url_finder.get_lane_count_data()
            self.video_source_url_finder.get_wrong_direction()
            self.video_source_url_finder.get_stopped_data()

            if self.is_lane_info_updated():
                print("Lanes information Updated")
                lanes = self.video_source_url_finder.get_configs()
             
            if not new_video_url:
                print("Failed to retrieve video URL!")
                break

            if new_video_url != self.current_video_url:
                print(f"Video URL updated: {new_video_url}")
                self.current_video_url = new_video_url

                self.cap = cv2.VideoCapture(self.current_video_url)
                if not self.cap.isOpened():
                    print("Failed to open video!")
                    break

            ret, frame = self.cap.read()
            if not ret:
                print("Video stream ended or failed to read.")
                continue
            
            self.get_video_properties()
            self.zone_manager.lanes_info(lanes)
            self.yolo_detector.stream_video_with_detections_and_tracker(frame, self.loaded_model, self.model)
            self.zone_manager.zone_counter(frame, self.yolo_detector.bbox_xyxy, self.yolo_detector.obj_ids)
            frame = self.drawer.draw_rectangle_to_image(frame, self.yolo_detector.bbox_xyxy, self.yolo_detector.obj_ids, self.yolo_detector.class_ids, self.model, self.yolo_detector.stopped_objects)
            self.zone_manager.draw_line(frame)
            zone_manager.send_lane_counts()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                break

if __name__ == "__main__":

    api_url = 'http://127.0.0.1:5001/get_video_url'
    configs_url = 'http://127.0.0.1:5001/get_config'
    data_url = "http://127.0.0.1:5001/get_lane_count_data"
    false_object_url = "http://127.0.0.1:5001/get_wrong_direction_object"
    stopped_object_url = "http://127.0.0.1:5001/get_stopped_object"
    model_name = "yolov7.pt"
    
    video_source_url_finder = VideoURLFinder(api_url, configs_url, data_url, false_object_url, stopped_object_url)
    zone_manager = ZoneManager()
    processor = VideoProcessor(video_source_url_finder, zone_manager, model_name)
    processor.main()
