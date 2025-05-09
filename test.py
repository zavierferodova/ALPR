import cv2
import os
from alpr.alpr import ALPR
from run import get_device, write_image
import pandas as pd

def scan_images(directory: str):
    image_extensions = ['.jpg', '.jpeg', '.png', '.wepb']
    image_files = []

    for data in os.scandir(directory):
        if data.is_file() and data.path.endswith(tuple(image_extensions)):
            image_files.append(data.path)

    return image_files

def test(base_path: str):
    image_files = scan_images(base_path)
    alpr = ALPR(device=get_device())

    output_path = f"{base_path}/output"
    correct_path = f"{output_path}/correct"
    incorrect_path = f"{output_path}/incorrect"

    df = pd.DataFrame(columns=['Actual', 'Predicted', 'Correct'])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(correct_path):
        os.mkdir(correct_path)
    
    if not os.path.exists(incorrect_path):
        os.mkdir(incorrect_path)

    total = len(image_files)
    correct = 0

    for image_file in image_files:
        image_name = os.path.basename(image_file)
        image = cv2.imread(image_file)
        x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
        results = alpr.predict(image)

        is_correct = False
        last_result = ""
        invalid_results = [None, "", " "]

        for result in results:
            if result[2] not in invalid_results:
                last_result = result[2]

            if result[2] == str(image_name.split('.')[0]):
                write_image(image, (x1, y1, x2, y2), f"{correct_path}/{image_name}")
                correct += 1
                is_correct = True
                break
        
        df_new = pd.DataFrame({'Actual': [image_name.split('.')[0]], 'Predicted': [last_result], 'Correct': [is_correct]})
        df = pd.concat([df, df_new], ignore_index=True)
        if not is_correct:
            write_image(image, (x1, y1, x2, y2), f"{incorrect_path}/{image_name}")

    df.to_csv('test-output.csv', index=False)
    percentage = correct/len(image_files) * 100
    print()
    print(f"Total images   : {total}")
    print(f"Accuracy (num) : {correct}/{total}")
    print(f"Accuracy ($%)  : {percentage:.2f}%")

if __name__ == '__main__':
    test("images")