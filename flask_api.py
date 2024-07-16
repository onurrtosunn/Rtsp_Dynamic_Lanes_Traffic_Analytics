from flask import Flask, request, jsonify

app = Flask(__name__)

class VideoURLManager:
    def __init__(self):
        self.video_url = "rtsp://localhost:9090/mystream"
        self.lanes = [
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
            },
            
        ]
        self.lane_count_data = None
        self.wrong_direction_object_data = None
        self.stopped_object_data = None

    def update_video_url(self, new_url):
        self.video_url = new_url

    def update_variable(self, new_config):
         self.lanes = new_config

    def get_video_url(self):
        return self.video_url
    
    def get_variable(self):
        return self.lanes

    def set_data(self, new_data):
        self.lane_count_data = new_data

    def set_false_data(self, new_data):
        self.wrong_direction_object_data = new_data
    
    def set_stopped_data(self, new_data):
        self.stopped_object_data = new_data

    def get_lane_count(self):
        return self.lane_count_data

    def get_wrong_direction_object(self):
        return self.wrong_direction_object_data
    
    def get_stopped_data(self):
        return self.stopped_object_data

video_manager = VideoURLManager()

@app.route('/update_video_url', methods=['PUT'])
def update_video():
    if 'new_url' in request.json:
        video_manager.update_video_url(request.json['new_url'])
        return jsonify({'message': 'Video URL updated successfully', 'new_url': video_manager.get_video_url()}), 200
    else:
        return jsonify({'error': 'Missing new_url parameter'}), 400

@app.route('/update_config', methods=['PUT'])
def update_config():
    try:
        data = request.get_json()
        lane_index = data.get('lane_index')
        key = data.get('key')
        value = data.get('value')

        if lane_index is not None and 0 <= lane_index < len(video_manager.lanes):
            if key in video_manager.lanes[lane_index]:
                video_manager.lanes[lane_index][key] = value
                return jsonify({'message': f'Config updated successfully for lane {lane_index}', 'lane': video_manager.lanes[lane_index]}), 200
            else:
                return jsonify({'error': f'Unknown key parameter: {key}'}), 400
        else:
            return jsonify({'error': f'Invalid lane_index parameter: {lane_index}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/send_lane_counts', methods=['PUT'])
def send_lane_counts():
    data = request.get_json()
    video_manager.set_data(data)
    return jsonify({"message": "Lane counts received successfully"}), 200

@app.route('/send_wrong_direction_object', methods=['PUT'])
def send_wrong_direction_object():
    data = request.get_json()
    video_manager.set_false_data(data)
    return jsonify({"message": "Wrong direction object information received successfully"}), 200

@app.route('/send_stopped_object', methods=['PUT'])
def send_stopped_object():
    data = request.get_json()
    video_manager.set_stopped_data(data)
    return jsonify({"message": "Stopped object information received successfully"}), 200

@app.route('/get_video_url', methods=['GET'])
def get_video_url():
    return jsonify({'video_url': video_manager.get_video_url()}), 200

@app.route('/get_config', methods=['GET'])
def get_variable():
    return jsonify({'config': video_manager.get_variable()}), 200

@app.route('/get_lane_count_data', methods=['GET'])
def get_lane_count_data():
    print("Sended Lane Information\n",video_manager.get_lane_count(),"\n")
    return jsonify({'data': video_manager.get_lane_count()}), 200

@app.route('/get_wrong_direction_object', methods=['GET'])
def get_wrong_direction():
    print("Sended Wrong Direction Information\n",video_manager.get_wrong_direction_object(),"\n")
    return jsonify({'false': video_manager.get_wrong_direction_object()}), 200

@app.route('/get_stopped_object', methods=['GET'])
def get_stop_direction():
    print("Sended Stopped Object Information\n", video_manager.get_stopped_data(), "\n")
    return jsonify({'stopped': video_manager.get_stopped_data()}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
