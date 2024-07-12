# Ayvos Case Study Usage

## Create RTSP Stream

```bash

./rtsp-simple-server
ffmpeg -re -stream_loop -1 -i Traffic.mp4 -f rtsp -rtsp_transport tcp rtsp://localhost:9090/mystream
```

## Run Python Files


Firtly run flask_api.py and than run vehicle_counter.py.
```bash

python3 flask_api.py
python3 vehicle_counter.py
```


Default configurations are specified in flask_api.py. if you want to make changes with config you can use the following command

```bash

curl -X PUT http://127.0.0.1:5001/update_config -H "Content-Type: application/json" -d '{"example_key":example_value, "example_key_2":example_value_2}'

```

## Reference 

https://github.com/bluenviron/mediamtx
