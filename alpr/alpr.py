import torch
import random
import cv2
import pandas as pd
import asyncio
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from cv2.typing import MatLike

class ALPR:
    def __init__(self, device: torch.device = "cpu"):
        self.vh_model = YOLO('models/yolo11s.pt')
        self.lp_model = YOLO('models/license_plate_model.pt')
        self.ap_model = YOLO('models/alphanumerics_plate_model.pt')
        self.vh_model.to(device)
        self.lp_model.to(device)
        self.ap_model.to(device)
        yolo_target_classes = ['car', 'truck', 'bus', 'motorcycle']
        self.vehicle_class_ids = [class_id for class_id, class_name in self.vh_model.names.items() if class_name in yolo_target_classes]

    def __detect_vehicles(self, frame: MatLike):
        """Detect vehicles from a frame."""
        results = self.vh_model(frame, conf=0.4, iou=0.5, classes=self.vehicle_class_ids, max_det=5, stream=True, half=True, agnostic_nms=True)
        for result in results:
            boxes = result.boxes
        return boxes

    def __detect_license_plates(self, frame: MatLike):
        """Detect license plates from a vehicle."""
        results = self.lp_model(frame, conf=0.4, iou=0.6, stream=True, half=True, max_det=1, agnostic_nms=True)
        for result in results:
            boxes = result.boxes
        return boxes
    
    def __detect_character_plates(self, frame: MatLike):
        """Detect alphanumeric characters from a license plate."""
        results = self.ap_model(frame, conf=0.4, iou=0.3, stream=True, half=True, agnostic_nms=True)
        for result in results:
            boxes = result.boxes
        return boxes
    
    def __cvrandom_color(self):
        """Generate a random color for OpenCV."""
        return [random.randint(0, 255) for _ in range(3)]
    
    def __filter_charplate(self, detected_chars: list[str], coords: list[tuple[int, int, int, int]]):
        """Filter detected chars based on average bounding box area"""

        if len(detected_chars) == 1:
            return detected_chars[0], coords

        # Calculate the average bounding box area
        areas = []
        for x1, y1, x2, y2 in coords:
            areas.append((x2 - x1) * (y2 - y1))
        avg_area = sum(areas) / len(areas)

        zip_data = zip(detected_chars, areas, coords)

        # Sort the data by area
        sorted_data = sorted(zip_data, key=lambda x: x[1])
        
        # Filter the data based on the IQR method
        areas = [area for _, area, _ in sorted_data]
        q1 = pd.Series(areas).quantile(0.25)
        q3 = pd.Series(areas).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3.5 * iqr
        upper_bound = q3 + 3.5 * iqr

        filtered_chars = []
        filtered_coords = []
        for char, area, coord in sorted_data:
            if lower_bound <= area <= upper_bound:
                filtered_chars.append(char)
                filtered_coords.append(coord)

        return filtered_chars, filtered_coords

    def __cluster_sort_charplate(self, detected_chars: list[str], coords: list[tuple[int, int, int, int]]):
        """
        Sort characters on a license plate image considering stacked characters.
        """
        if len(detected_chars) == 0:
            return ""
        
        if len(detected_chars) == 1:
            return detected_chars[0]

        # Create a DataFrame from detected texts and coordinates
        df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2'])
        df['text'] = detected_chars

        # Calculate the centroid of each bounding box
        df['cx'] = (df['x1'] + df['x2']) / 2
        df['cy'] = (df['y1'] + df['y2']) / 2

        # Step 1: Cluster rows by y-coordinates (using a simple threshold or k-means)
        clustering_threshold = 10  # Adjust based on the character size and spacing
        df = df.sort_values(by='cy')
        row_groups = []
        current_row = [df.iloc[0]]

        for i in range(1, len(df)):
            if abs(df.iloc[i]['cy'] - current_row[-1]['cy']) <= clustering_threshold:
                current_row.append(df.iloc[i])
            else:
                row_groups.append(pd.DataFrame(current_row))
                current_row = [df.iloc[i]]
        row_groups.append(pd.DataFrame(current_row))

        # Step 2: Sort each row by x-coordinates and merge
        sorted_texts = []
        for row in row_groups:
            row = row.sort_values(by='cx')
            sorted_texts.extend(row['text'].tolist())

        return " ".join(sorted_texts)

    def __inner_coords(self, box: Boxes, crop_coords: tuple[int, int, int, int]|None = None):
        """Get the inner coordinates of a box."""
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if crop_coords is not None:
            x1_1, y1_1, x2_2, y2_2 = crop_coords
            x1 += x1_1
            y1 += y1_1
            x2 += x1_1
            y2 += y1_1
        return (x1, y1, x2, y2)
    
    def __draw_label_box(
        self,
        frame: MatLike,
        model: YOLO,
        box: Boxes, 
        color: tuple[int, int, int],
        coords: tuple[int, int, int, int]|None = None
    ) -> tuple[int, int, int, int]:
        """Draw a label box on the frame."""
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = box.conf.item()

        x1, y1, x2, y2 = coords

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare the text
        label = f"{class_name}: {confidence:.2f}"
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Create a background rectangle for the text
        text_x = x1
        text_y = y1 - 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), color, cv2.FILLED)

        # Draw the text
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    def __draw_plate_number(self, frame: MatLike, text: str, coords: tuple[int, int]):
        """Draw the detected license plate text on the frame."""
        x, y = coords

        text_x = x
        text_y = y + 15
        font_scale = 0.3
        font_thickness = 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Create a background rectangle for the text
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (255, 0, 0), cv2.FILLED)
        
        # Draw the text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)\
        
    async def __char_detection(self, frame, license_coords: tuple[int, int, int, int]):
        """Detect alphanumeric characters from a license plate."""
        # Forward the cropped license plate image to the alphanumerics plate model
        lp_x1, lp_y1, lp_x2, lp_y2 = license_coords
        lp_crop = frame[lp_y1:lp_y2, lp_x1:lp_x2]

        # Detect box for alphanumeric characters from the license plate
        ap_boxes = self.__detect_character_plates(lp_crop)
        
        detected_texts = []
        detected_coords = []
        if len(ap_boxes) != 0:
            for alpha_box in ap_boxes:
                alpha_x1, alpha_y1, alpha_x2, alpha_y2 = self.__inner_coords(alpha_box, (lp_x1, lp_y1, lp_x2, lp_y2))

                # Store detected text and its position
                text_class_id = int(alpha_box.cls.item())
                detected_texts.append(self.ap_model.names[text_class_id])
                detected_coords.append((alpha_x1, alpha_y1, alpha_x2, alpha_y2))

            # Filter and sort the detected characters
            # filtered_chars, filtered_coords = self.__filter_charplate(detected_texts, detected_coords)
            final_text = self.__cluster_sort_charplate(detected_texts, detected_coords)

            # Draw the detected alphanumeric characters
            # for x1, y1, x2, y2 in detected_coords:
            #     self.__draw_label_box(frame, self.ap_model, alpha_box, (255, 0, 0), (x1, y1, x2, y2))

            # Write detected text to file if any text was detected
            if final_text:
                # Draw the detected text below the license plate rectangle
                self.__draw_plate_number(frame, final_text, (lp_x1, lp_y2))

            return final_text

        return None

    async def __license_detection(self, frame: MatLike, vehicle_coords: tuple[int, int, int, int]):
        """Detect license plates from a vehicle."""
        # Detect license plates from the vehicle
        x1, y1, x2, y2 = vehicle_coords
        vehicle_crop = frame[y1:y2, x1:x2]
        lp_boxes = self.__detect_license_plates(vehicle_crop)

        lp_coords = None
        char_plate = None
        if len(lp_boxes) != 0:
            for lp_box in lp_boxes:
                # Get the inner coordinates of the license plate box
                lp_x1, lp_y1, lp_x2, lp_y2 = self.__inner_coords(lp_box, (x1, y1, x2, y2))
                
                # Draw license plate label box
                self.__draw_label_box(frame, self.lp_model, lp_box, (0, 255, 0), (lp_x1, lp_y1, lp_x2, lp_y2))

                # Store the license plate coordinates
                lp_coords = (lp_x1, lp_y1, lp_x2, lp_y2)

                # Process alphanumeric characters
                char_plate = await self.__char_detection(frame, license_coords=(lp_x1, lp_y1, lp_x2, lp_y2))

        return lp_coords, char_plate

    async def __forward(self, frame: MatLike):
        """Forward predictions to the models."""
        # Detect vehicles
        boxes = self.__detect_vehicles(frame)

        vh_coords = []
        processes = []
        if len(boxes) != 0:
            for box in boxes:
                # Get the inner coordinates of the box
                x1, y1, x2, y2 = self.__inner_coords(box)

                # Draw vehicle label box
                self.__draw_label_box(frame, self.vh_model, box, self.__cvrandom_color(), (x1, y1, x2, y2))

                # Store the vehicle coordinates
                vh_coords.append((x1, y1, x2, y2))

                # Process license plate and alphanumeric characters
                processes.append(self.__license_detection(frame, vehicle_coords=(x1, y1, x2, y2)))

        processes = tuple(processes)
        responses = await asyncio.gather(*processes)

        results = []
        for vh_coord, (lp_coords, char_plates) in zip(vh_coords, responses):
            results.append((vh_coord, lp_coords, char_plates))
        return results
            
    def predict(self, frame: MatLike):
        """Predict license plates and alphanumeric characters from a frame."""
        responses = asyncio.run(self.__forward(frame))
        return responses
