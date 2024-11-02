import cv2
import torch
from ultralytics import YOLO
from djitellopy import Tello
import time
import pandas as pd
from datetime import datetime

def initialize_tello():
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()  # Start the video streaming
    time.sleep(2)  # Give the stream time to start
    tello.set_speed(10)
    return tello

def record_tello_video_stream(out: cv2.VideoWriter, tello: Tello, frame_read):
    """
    Records video from Tello while performing other tasks.
    """
    # Write the captured frame to the output video file
    frame = frame_read.frame
    out.write(frame)

    # Display the captured frame in a window
    # cv2.imshow("Tello Object Navigation", frame)

def main():
    tello = initialize_tello()
    
    # Setup YOLOv5 for object detection
    model = YOLO('yolov8s-world.pt')
    
    # Define the target object type to track
    target_object = 'teddy bear'  # Change this to track different objects
    classes = [target_object]
    class_dict = {}
    for i, c in enumerate(classes):
        class_dict[i] = c
    
    model.set_classes(classes)
 
    # Constants for drone control
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 720
    FRAME_CENTER = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
    OBJECT_CENTER_TOLERANCE = 50  # Pixel tolerance for centering
    OBJECT_TARGET_AREA = 15000  # Desired area for object to maintain consistent distance
    FLIGHT_DURATION = 75  # Flight duration in seconds (1.5 minutes)

    # Initialize video writer for recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    frame_read = tello.get_frame_read()
    H, W, _ = frame_read.frame.shape
    out = cv2.VideoWriter(f'{target_object}_{timestamp}_yoloworld_recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (W, H))

    # Initialize timing variables
    start_time = time.time()
    last_action_time = start_time
    tello.takeoff()
    print(tello.get_battery())
    
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            print("Failed to grab frame")
            continue

        # Resize frame as soon as it is read
        resized_frame = cv2.resize(frame, (640, 480))
          
        current_time = time.time()

        # Check if the flight duration has exceeded the limit
        if current_time - start_time >= FLIGHT_DURATION:
            break

        # Perform drone movement and object detection every 1.0 second to keep it interleaved with movement
        if current_time - last_action_time >= 1.0:
            # Object detection logic
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            
            # Filter detections for the target object
            box_cls = results[0].boxes.cls
            box_conf = results[0].boxes.conf
            box_xyxy = results[0].boxes.xyxy

            detections = pd.DataFrame(box_xyxy.numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
            detections['confidence'] = box_conf
            detections['class'] = box_cls
            detections['name'] = [class_dict[i] for i in box_cls.int().tolist()]

            target_detections = detections[detections['name'] == target_object]

            if not target_detections.empty:
                # Get the largest detection (assuming the closest object)
                largest_detection = target_detections.iloc[0]
                object_x_center = (largest_detection['xmin'] + largest_detection['xmax']) / 2
                object_y_center = (largest_detection['ymin'] + largest_detection['ymax']) / 2
                object_area = (largest_detection['xmax'] - largest_detection['xmin']) * (largest_detection['ymax'] - largest_detection['ymin'])

                # Check if the object is centered before initiating orbit
                if (abs(object_x_center - FRAME_CENTER[0]) <= OBJECT_CENTER_TOLERANCE and
                        abs(object_y_center - FRAME_CENTER[1]) <= OBJECT_CENTER_TOLERANCE):
                    
                    # Adjust drone position to maintain circular orbit with radius of 1.5m
                    tello.move_right(261)  # Move right to create the orbiting motion, distance calculated per segment
                    tello.rotate_counter_clockwise(10)  # Rotate counter clockwise by 10 degrees to keep the object in the center

                # Adjust drone position to keep the object in the center of the frame
                if object_x_center < FRAME_CENTER[0] - OBJECT_CENTER_TOLERANCE:
                    tello.move_right(20)  # Object is to the left, move right to recenter
                elif object_x_center > FRAME_CENTER[0] + OBJECT_CENTER_TOLERANCE:
                    tello.move_left(20)  # Object is to the right, move left to recenter

                if object_y_center < FRAME_CENTER[1] - OBJECT_CENTER_TOLERANCE:
                    tello.move_up(20)  # Object is above, move up to recenter
                elif object_y_center > FRAME_CENTER[1] + OBJECT_CENTER_TOLERANCE:
                    tello.move_down(20)  # Object is below, move down to recenter

                # Adjust distance to the target to maintain consistent following distance
                if object_area > OBJECT_TARGET_AREA:
                    tello.move_back(20)  # Object too close, move back
                elif object_area < OBJECT_TARGET_AREA:
                    tello.move_forward(20)  # Object too far, move forward

            last_action_time = current_time  # Update last action time

        # Dynamic bounding box for the target object while drone is rotating
        if not target_detections.empty:
            for idx, detection in target_detections.iterrows():
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                label = f"{detection['name']} {detection['confidence']:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Tello Object Navigation', resized_frame)
        # Record the video stream using the defined function
        record_tello_video_stream(out, tello, frame_read)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Cleanup
    tello.streamoff()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()
    tello.land()

if __name__ == '__main__':
    main()
