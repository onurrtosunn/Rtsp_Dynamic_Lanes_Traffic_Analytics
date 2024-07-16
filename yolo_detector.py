import cv2
import numpy as np
from PIL import Image
from sort import Sort
from collections import defaultdict

class YOLODetection:
    def __init__(self):
         
        self.initialize_background_subtraction()
        self.initialize_sort_tracker()
        self.stopped_objects = defaultdict(int)

    def initialize_background_subtraction(self):

        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    def initialize_sort_tracker(self):

        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    def perform_inference(self, frame, loaded_model):

        converted_img = Image.fromarray(frame[:, :, ::-1], 'RGB')
        detection_object = loaded_model(converted_img)
        detection_result = detection_object.xyxy[0].cpu().detach().numpy()
        fg_mask = self.back_sub.apply(frame)
        return self.analyzer(fg_mask, detection_result)

    def analyzer(self, fg_mask, detection_result, confidence_threshold=0.51):

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

    def track_objects_with_sort(self, detections):
        
        return self.sort_tracker.update(detections)

    def stream_video_with_detections_and_tracker(self,frame, loaded_model, model):

        self.model = model
        detections = self.perform_inference(frame, loaded_model)
        tracked_detections = self.track_objects_with_sort(detections)
        if len(tracked_detections) > 0:
            self.parse_drawing_information(tracked_detections)

    def parse_drawing_information(self, detections):

        self.bbox_xyxy = detections[:, :4]
        self.obj_ids = detections[:, -1].astype(int)
        self.class_ids = detections[:, 4].astype(int)