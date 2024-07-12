# Ayvos

## Create RTSP Stream

```bash

./rtsp-simple-server
ffmpeg -re -stream_loop -1 -i Traffic.mp4 -f rtsp -rtsp_transport tcp rtsp://localhost:9090/mystream
```

## Reference 

https://github.com/bluenviron/mediamtx
