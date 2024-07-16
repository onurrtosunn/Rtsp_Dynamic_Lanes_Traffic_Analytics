import cv2
import requests
import numpy as np
from PIL import Image
from collections import defaultdict

class Drawer:
    def __init__(self):
        
        self.stopped_threshold = 3
        self.background_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.stopped_objects = defaultdict(int) 
        self.line_thickness = 3

    def draw_rectangle_to_image(self, img, bbox, obj_ids, class_ids, model, stopped_objects):

        for bbox, obj_id, class_id in zip(bbox, obj_ids, class_ids):
            x1, y1, x2, y2 = [int(i) for i in bbox]

            label = f"ID: {obj_id} - Class: {model.names[class_id]}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.line_thickness)
            text_position = (x1, y1 - 2)

            obj_coords = (x1, y1, x2, y2)
            if stopped_objects.get(obj_coords, 0) >= self.stopped_threshold:
                bbox_color = (0, 0, 255)
                self.send_stopped_object(obj_id)
            else:
                bbox_color = (0, 255, 0) 

            cv2.rectangle(img, (x1, y1 - text_size[1] - 3), (x1 + text_size[0], y1 - 2), self.background_color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.text_color, 1)
        return img

    def send_stopped_object(self, obj_id):

        try:
            text = f"{obj_id} Stopped !"
            response = requests.put("http://127.0.0.1:5001/send_stopped_object", json=text)

            if response.status_code == 200:
                print(text)
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending data: {e}")




            