# Ayvos Case Study Usage

## Create RTSP Stream

```bash

./rtsp-simple-server
ffmpeg -re -stream_loop -1 -i Traffic.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -keyint_min 30 -f rtsp -rtsp_transport tcp rtsp://localhost:9090/mystream
```

## Run Python Files


Firtly run flask_api.py and than run vehicle_counter.py.
```bash

python3 flask_api.py
python3 vehicle_counter.py
```


Default configurations are specified in flask_api.py. if you want to make changes with config you can use the following command

### To change lanes information
```bash

curl -X PUT http://127.0.0.1:5001/update_config  -H "Content-Type: application/json"  -d '{"lane_index": {selected_lane}, "key": "{selected_line}", "value": [{x_value}, {y_value}]}'

```
#### For example

```bash
curl -X PUT http://127.0.0.1:5001/update_config  -H "Content-Type: application/json"  -d '{"lane_index": 5, "key": "line_1_start", "value": [830, 446]}'
```

### To change rtsp source
```bash
curl -X PUT http://127.0.0.1:5001/update_video_url -H "Content-Type: application/json" -d '{"new_url":"{new_rtsp stream or video path}"}'
```
#### For example

```bash
curl -X PUT http://127.0.0.1:5001/update_video_url -H "Content-Type: application/json" -d '{"new_url":"rtsp://localhost:9090/mystream"}'
```

## RTSP open ports
- [RTSP] UDP/RTP listener opened on :8000

- [RTSP] UDP/RTCP listener opened on :8001

- [RTSP] TCP listener opened on :9090

- [RTMP] listener opened on :1935

- [HLS] listener opened on :8888

## Notes 
- Stopped objects are coloured with a red rectangle and a threshold value is used to prevent false positives.
- createBackgroundSubtractorMOG2 algorithm is used to detect non-moving objects
- Sort algorith is used to track each objecet 
- Vehicle counts, stops and wrong-way vehicle information are sent immediately to the endpoint.
- A demo version of the video has been uploaded to github repo
- Parallel stream analysis is not supported by the code.

## Docker


```bash
docker run --rm --name ayvos -it -p 9090:9090 -p 8555:8555 -p 8000:8000/udp -p 8001:8001/udp -p 1935:1935 -p 8888:8888 -w /Ayvos/ -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix onurrtosunn/ayvos_case_study:v1

```

## Screenshots
![image](https://github.com/user-attachments/assets/185924a7-2023-4757-87cf-c8966fc16e18)
![image](https://github.com/user-attachments/assets/fbe3a8ab-cfbe-4baa-9e2c-dc4667f03be1)

## Reference 

https://github.com/bluenviron/mediamtx
