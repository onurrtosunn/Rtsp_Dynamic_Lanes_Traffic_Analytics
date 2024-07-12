from flask import Flask, request, jsonify

app = Flask(__name__)

class VideoURLManager:
    def __init__(self):
        self.video_url = "Traffic.mp4"
        self.config = {
            'line1_x1': 5,
            'line1_x2': 10,
            'line1_y1': 8,
            'line1_y2': 6,
            'start_point_1': 52,
            'end_point_1': 80,
            'line2_x1': 51,
            'line2_x2': 30,
            'line2_y1': 18,
            'line2_y2': 13,
            'start_point_2': 42,
            'end_point_2': 30
        }

    def update_video_url(self, new_url):
        self.video_url = new_url

    def update_variable(self, new_config):
         self.config = new_config

    def get_video_url(self):
        return self.video_url
    
    def get_variable(self):
        return self.config

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
        for key, value in data.items():
            if key in video_manager.config:
                video_manager.config[key] = value
            else:
                return jsonify({'error': f'Unknown config parameter: {key}'}), 400
        return jsonify({'message': 'Config updated successfully', 'config': video_manager.config}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_video_url', methods=['GET'])
def get_video_url():
    return jsonify({'video_url': video_manager.get_video_url()}), 200

@app.route('/get_config', methods=['GET'])
def get_variable():
    return jsonify({'config': video_manager.get_variable()}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
