import cv2
import torch
from ultralytics import YOLO
from djitellopy import Tello
import time
import pandas as pd

def initialize_tello():
    # Initialize and connect the Tello drone
    tello = Tello('192.168.87.34')
    tello.connect()
    print(tello.get_battery())
    # Start video streaming
    tello.streamon()
    time.sleep(2)  # Allow time for the video stream to start
    tello.set_speed(10)
    return tello

def main():
    tello = initialize_tello()
    
    # Setup YOLOv5 for object detection
    model = YOLO('yolov8s-world.pt')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Define the target object type to track
    target_object = 'person'  # Change this to track different objects
    classes = ["person"]
    class_dict = {}
    for i, c in enumerate(classes):
        class_dict[i] = c
    
    model.set_classes(classes)
 
    # Constants for drone control
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 720
    FRAME_CENTER = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
    OBJECT_TARGET_AREA = 80000  # Adjust based on trial to maintain desired distance
    OBJECT_CENTER_TOLERANCE = 50  # Pixel tolerance for centering
    state = "Not Centered"

    # Initialize timing variables
    last_action_time = time.time()
    
    # Initialize frame variables
    largest_detection = None
    
    tello.takeoff()
    print(tello.get_battery())
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            print("Failed to grab frame")
            continue
    
        current_time = time.time()

        # Perform object detection and movement every 2 seconds
        if current_time - last_action_time >= 2:
            # Object detection logic
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            
            # Filter detections for the target object
            box_cls = results[0].boxes.cls
            box_conf = results[0].boxes.conf
            box_xyxy = results[0].boxes.xyxy

            detections = pd.DataFrame(box_xyxy.numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
            detections['confidence'] = box_conf
            detections['class'] = box_cls
            detections['name'] = [class_dict[i] for i in box_cls.int().tolist()]

            print(detections.head(5))
            target_detections = detections[detections['name'] == target_object]
            # print("Target Detection", target_detections)

            if not target_detections.empty:
                largest_detection = target_detections.iloc[0]
                object_x_center = (largest_detection['xmin'] + largest_detection['xmax']) / 2
                # print("Object X Center is: ", object_x_center)
                object_y_center = (largest_detection['ymin'] + largest_detection['ymax']) / 2
                # print("Object Y Center is: ",object_y_center)
                object_area = (largest_detection['xmax'] - largest_detection['xmin']) * (largest_detection['ymax'] - largest_detection['ymin'])
                print("Object area is: ", object_area)

                state = "CENTERED"
                # tello.move_right(20)
                # tello.rotate_counter_clockwise(15)
                # time.sleep(1)
                
                # Adjust drone position to center the object
                if object_x_center < FRAME_CENTER[0] - OBJECT_CENTER_TOLERANCE:
                    tello.move_left(20)
                    tello.rotate_clockwise(10)
                elif object_x_center > FRAME_CENTER[0] + OBJECT_CENTER_TOLERANCE:
                    tello.move_right(60)
                    tello.rotate_counter_clockwise(25)
                    tello.move_right(30)
                    tello.rotate_counter_clockwise(20)
                    tello.move_right(40)
                    tello.rotate_counter_clockwise(25)
                    tello.move_right(20)
                    # tello.rotate_counter_clockwise(10)

                if object_y_center < FRAME_CENTER[1] - OBJECT_CENTER_TOLERANCE:
                    tello.move_up(20)
                elif object_y_center > FRAME_CENTER[1] + OBJECT_CENTER_TOLERANCE:
                    tello.move_down(20)

                # Adjust drone's distance to maintain desired following distance
                if object_area > OBJECT_TARGET_AREA:
                    tello.move_back(20)
                elif object_area < OBJECT_TARGET_AREA:
                    tello.move_forward(20)
            
    
            last_action_time = current_time  # Update the last action time
    
        if largest_detection is not None:
            x1, y1, x2, y2 = int(largest_detection['xmin']), int(largest_detection['ymin']), int(largest_detection['xmax']), int(largest_detection['ymax'])
            label = f"{largest_detection['name']} {largest_detection['confidence']:.2f}"
            color = (0, 255, 0)  # Green box for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Tello Object Navigation', frame)
    
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    
    # Cleanup
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.land()

if __name__ == '__main__':
    main()