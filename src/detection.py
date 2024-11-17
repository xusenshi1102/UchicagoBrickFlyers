from ultralytics import YOLO
import pandas as pd
import cv2


class YOLODetector:
    """Class to handle object detection using YOLO."""

    def __init__(self, model_path, target_object):
        """Initialize the YOLO detector.

        Args:
            model_path (str): Path to the YOLO model.
            target_object (str): Name of the target object to detect.
        """
        self.model = YOLO(model_path)
        self.target_object = target_object
        self.classes = [target_object]
        self.class_dict = {i: c for i, c in enumerate(self.classes)}

    def detect(self, frame):
        """Detect objects in the frame.

        Args:
            frame (numpy.ndarray): The image frame in which to detect objects.

        Returns:
            pandas.DataFrame: DataFrame containing detections of the target object.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        boxes = results[0].boxes

        box_cls = boxes.cls
        box_conf = boxes.conf
        box_xyxy = boxes.xyxy

        detections = pd.DataFrame(
            box_xyxy.numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax']
        )
        detections['confidence'] = box_conf
        detections['class'] = box_cls
        detections['name'] = [
            self.class_dict.get(int(i), 'Unknown') for i in box_cls.int().tolist()
        ]
        target_detections = detections[detections['name'] == self.target_object]

        return target_detections
