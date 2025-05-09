import cv2
import torch
import os
import datetime
import threading
from cv2 import VideoCapture
from alpr.alpr import ALPR

def get_device():
    if torch.cuda.is_available():            # For Nvidia GPU
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():  # For Mac Silicon & AMD
        return torch.device("mps")
    else:
        return torch.device("cpu")

def write_text(text: str, filename: str):
    with open(filename, 'w') as f:
        f.write(text)

def write_image(frame, coords, filename):
    x1, y1, x2, y2 = coords
    cv2.imwrite(filename, frame[y1:y2, x1:x2])

def write_predictions(frame, results):
    base_path = "detections"

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    ct = datetime.datetime.now()
    str_ct = ct.strftime('%Y-%m-%d_%H.%M.%S.%f')

    single_path = f"{base_path}/{str_ct}"

    if not os.path.exists(single_path):
        os.mkdir(single_path)

        if len(results) > 0:
            for index, (vh_coords, lp_coords, char_plate) in enumerate(results):
                write_image(frame, vh_coords, f"{single_path}/vehicle-{index}.jpg")

                if lp_coords is not None:
                    write_image(frame, lp_coords, f"{single_path}/license_plate-{index}.jpg")

                if char_plate is not None:
                    write_text(char_plate, f"{single_path}/char_plate-{index}.txt")

def run_model(
        video_path: str,
        skip_frames: int = 1,
        width_resize: int|None = None,
        display_frame: bool = True,
        write_results: bool = False,
        write_interval: int = 5,
        export: str|None = None
    ):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    alpr = ALPR(device=get_device())
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    last_write_time = datetime.datetime.now()

    if export is not None:
        video_writer = cv2.VideoWriter(export, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        if frame_count == fps:
            frame_count = 0

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        # Resize the frame to a width while maintaining aspect ratio
        if width_resize is not None:
            height, width = frame.shape[:2]
            new_width = width_resize
            new_height = int((new_width / width) * height)
            frame = cv2.resize(frame, (new_width, new_height))

        copy_frame = frame.copy()
        results = alpr.predict(frame)

        current_time = datetime.datetime.now()
        if write_results and (current_time - last_write_time).total_seconds() >= write_interval:
            threading.Thread(target=write_predictions, args=(copy_frame, results)).start()
            last_write_time = current_time
            
        if (export is not None):
            video_writer.write(frame)

        if (display_frame):
            # Show the frame with vehicle, license plate, and alphanumeric detections
            cv2.imshow('ALPR', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Run the model
    run_model("videos/test.mp4", 
        skip_frames=6,
        write_results=False,
        write_interval=3,
        # export="videos/output/test_output.mp4"
    )
