import cv2
import requests
import time
import signal
import sys

class VideoURLFinder:
    def __init__(self, video_api_url, variable_url):
        self.video_api_url = video_api_url
        self.variable_url = variable_url

    def get_video_url(self):
        response = requests.get(self.video_api_url)
        if response.status_code == 200:
            return response.json()['video_url']
        else:
            return None
    
    def get_variable(self):
        response = requests.get(self.variable_url)
        if response.status_code == 200:
            return response.json()['variable']
        else:
            return None

class VideoProcessor:
    def __init__(self, video_source_url_finder):
        self.video_source_url_finder = video_source_url_finder
        self.current_video_url = None
        self.loop_count = 0

    def main(self):
        while True:
            new_video_url = self.video_source_url_finder.get_video_url()
            variable = self.video_source_url_finder.get_variable()

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
                time.sleep(1)
                continue

            self.loop_count += 1
            print(f"Loop count: {self.loop_count}")
            print(variable)
            print(self.current_video_url)

            cv2.imwrite("frame.jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                break
            time.sleep(1)

if __name__ == "__main__":
    api_url = 'http://127.0.0.1:5001/get_video_url'
    variable_url = 'http://127.0.0.1:5001/get_variable'
    video_source_url_finder = VideoURLFinder(api_url, variable_url)
    processor = VideoProcessor(video_source_url_finder)
    processor.main()
