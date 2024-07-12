import ffmpeg
import threading
import time
import logger

def start_rtsp_stream(input_file, port):
    try:
        (
            ffmpeg
            .input(input_file, stream_loop=-1)
            .output(f'rtsp://localhost:{port}/stream', format='rtsp', rtsp_transport='tcp')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        logger.print_with_timestamp("RTSP streaming has started successfully.")
    
    except KeyboardInterrupt:
        logger.print_with_timestamp("Streaming terminated by user.")
    
    except Exception as e:
        logger.print_with_timestamp(f"An unexpected error occurred: {e}")

def other_operations():
    # Perform other operations here
    logger.print_with_timestamp("Performing other operations...")
    time.sleep(5)  # Example: Wait for 5 seconds
    logger.print_with_timestamp("Other operations completed.")

if __name__ == "__main__":
    input_file = 'Traffic.mp4'
    port = 8554

    # Start RTSP stream in a separate thread
    rtsp_thread = threading.Thread(target=start_rtsp_stream, args=(input_file, port))
    rtsp_thread.start()

    # Continue with other operations concurrently
    other_operations()

    # Wait for the RTSP thread to complete (optional)
    rtsp_thread.join()

    logger.print_with_timestamp("All tasks completed.")
