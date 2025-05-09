# Automatic License Plate Recognition (ALPR)

The Python script is designed for real-time vehicle detection and license plate recognition, utilizing YOLOv11. It features a three-layer cascading model that performs forward predictions to extract license plate information. The application can take input from a video stream or video file facilitated through the OpenCV library.

List of contents :
- [Automatic License Plate Recognition (ALPR)](#automatic-license-plate-recognition-alpr)
  - [Usage](#usage)
  - [Function Params](#function-params)
  - [Testing](#testing)
    - [1. Setup](#1-setup)
    - [2. Change Arguments Input](#2-change-arguments-input)
    - [3. Running](#3-running)
    - [4. Result](#4-result)

## Usage
1. Install required depedencies
   ```sh
   pip install torch opencv-python pandas ultralytics
   ```
2. Make sure you already setup arguments for `run_model()` function in `run.py` script
   ```python
   # Example `run.py`

   if __name__ == '__main__':
      run_model("videos/test.mp4", 
        skip_frames=6,
        write_results=True,
        write_interval=3,
        export="videos/output/test_output.mp4"
      )
   ```
3. Hit and run
   ```sh
   python run.py
   ```
4. The OpenCV window will poup-up on your device screen and start playing your video input
5. Enjoyy !!

## Function Params

Below is a list of parameters that `run_model()` function receives:

- `video_path: str` - The input video file path location.
  
- `skip_frames: int` - By default, the application always predicts every single frame of video pics, to reduce computing resources you can change this parameter value.
  
- `width_resize: int|None` - Resize application window width while maintaining aspect ratio from video input.
  
- `display_frame: bool` - Show OpenCV window frame when application playing video input.
  
- `write_results: bool` - Output the result of predictions into files containing crop images of the vehicle, license plate, and character recognition stored in the "detections" folder.
  
- `write_interval: int` - The seconds interval result of predictions output will generated.

- `export: str|None` - Output file path of streamed video prediction result.

## Testing
Application accuracy is evaluated through image-level testing; if the application can correctly predict the full character results from license plates, then the results are considered accurate.

### 1. Setup
Create new directory and give label on the image with expected result of license plate information. Example: `A D 1 2 3 C D.jpg`

The directory tree will look like this :
```
images
├── A A 1 0 3 4 F X.jpg
├── A B 1 5 4 0 I J.jpg
├── A B 4 9 1 6 D A.jpg
├── A D 2 A.jpeg
├── A D 4 4 2 6 L B.jpg
└── W 4 3 9 7 B U.jpg
```

### 2. Change Arguments Input

After labeling the image open `test.py` file then make sure `test()` functions arguments input pointed into images directory.
```python
# Example `test.py`

if __name__ == '__main__':
    test("images")
```

### 3. Running
Just run `test.py` to start accuracy test and wait till finish.
```sh
python test.py
```

### 4. Result
When the test result is finish application will give output like shown below :

```
Total images   : 20
Accuracy (num) : 14/20
Accuracy ($%)  : 70.00%
```

On the other side, you can also review test results by opening `test-output.csv` files or viewing `output` directory that is generated inside the image test directory.

```
├── images
│   └── output
│       ├── correct
│       └── incorrect
└── test-output.csv
```