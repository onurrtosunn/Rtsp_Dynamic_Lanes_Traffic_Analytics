from flask import Flask, request, jsonify

app = Flask(__name__)

class VideoURLManager:
    def __init__(self):
        self.video_url = "Traffic.mp4"
        self.variable = None  # Başlangıçta değişkeni None olarak tanımladık

    def update_video_url(self, new_url):
        self.video_url = new_url

    def update_variable(self, new_variable):
        self.variable = new_variable

    def get_video_url(self):
        return self.video_url
    
    def get_variable(self):
        return self.variable

video_manager = VideoURLManager()

@app.route('/update_video_url', methods=['PUT'])
def update_video_url():
    if 'new_url' in request.json:
        video_manager.update_video_url(request.json['new_url'])
        return jsonify({'message': 'Video URL updated successfully', 'new_url': video_manager.get_video_url()}), 200
    else:
        return jsonify({'error': 'Missing new_url parameter'}), 400

@app.route('/update_variable', methods=['PUT'])
def update_variable():
    if 'variable' in request.json:
        video_manager.update_variable(request.json['variable'])
        return jsonify({'message': 'Video VARIABLE updated successfully', 'new_variable': video_manager.get_variable()}), 200
    else:
        return jsonify({'error': 'Missing variable parameter'}), 400

@app.route('/get_video_url', methods=['GET'])
def get_video_url():
    return jsonify({'video_url': video_manager.get_video_url()}), 200

@app.route('/get_variable', methods=['GET'])
def get_variable():
    return jsonify({'variable': video_manager.get_variable()}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
