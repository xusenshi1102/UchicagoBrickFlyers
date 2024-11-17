from navigation import TelloDrone, DroneError
from detection import YOLODetector
import os
import cv2
import datetime
import time

class DroneController:
    """Class to control the drone and object tracking."""

    def __init__(self, target_object, drone_ip, flight_duration=75, ):
        """Initialize the drone controller.

        Args:
            target_object (str): The object to track.
            flight_duration (int): Flight duration in seconds.
        """
        self.video_path = None
        self.drone = TelloDrone(drone_ip)
        self.detector = YOLODetector('yolov8s-world.pt', target_object)
        self.target_object = target_object
        self.flight_duration = flight_duration  # in seconds
        self.frame_width = 960
        self.frame_height = 720
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        self.object_center_tolerance = 50  # in pixels
        self.object_target_area = 15000
        self.last_action_time = 0
        self.start_time = 0
        self.out = None  # Video writer
        self.frame_read = None

    def initialize_video_writer(self):
        """Initialize the video writer for recording."""
        frame = self.drone.get_frame()
        if frame is not None:
            height, width, _ = frame.shape

            # Create directory if it doesn't exist
            video_dir = os.path.join('object', self.target_object, 'video')
            os.makedirs(video_dir, exist_ok=True)

            video_path = os.path.join(
                video_dir,
                f'{self.target_object}_yoloworld_recording.avi'
            )
            self.video_path = video_path
            self.out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                30,
                (width, height)
            )
        else:
            raise DroneError("Cannot initialize video writer without a frame.")

    def record_frame(self, frame):
        """Record the current frame to the video file.

        Args:
            frame (numpy.ndarray): The frame to record.
        """
        if self.out is not None:
            self.out.write(frame)

    def adjust_drone_position(self, detection):
        """Adjust the drone position based on the detection.

        Args:
            detection (pandas.Series): The detection data for the target object.
        """
        object_x_center = (detection['xmin'] + detection['xmax']) / 2
        object_y_center = (detection['ymin'] + detection['ymax']) / 2
        object_area = (detection['xmax'] - detection['xmin']) * (
            detection['ymax'] - detection['ymin']
        )

        # Adjust drone position to keep the object in the center of the frame
        try:
            if object_x_center < self.frame_center[0] - self.object_center_tolerance:
                self.drone.move_right(20)  # Object is to the left, move right to recenter
            elif object_x_center > self.frame_center[0] + self.object_center_tolerance:
                self.drone.move_left(20)  # Object is to the right, move left to recenter

            if object_y_center < self.frame_center[1] - self.object_center_tolerance:
                self.drone.move_up(20)  # Object is above, move up to recenter
            elif object_y_center > self.frame_center[1] + self.object_center_tolerance:
                self.drone.move_down(20)  # Object is below, move down to recenter

            # Adjust distance to the target to maintain consistent following distance
            if object_area > self.object_target_area:
                self.drone.move_back(20)  # Object too close, move back
            elif object_area < self.object_target_area:
                self.drone.move_forward(20)  # Object too far, move forward
        except Exception as e:
            print(f"Error adjusting drone position: {e}")

    def orbit_object(self):
        """Perform orbit around the object."""
        try:
            self.drone.move_right(261)
            self.drone.rotate_counter_clockwise(10)
        except Exception as e:
            print(f"Error during orbit: {e}")

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame.

        Args:
            frame (numpy.ndarray): The image frame to draw on.
            detections (pandas.DataFrame): The detections to draw.
        """
        for _, detection in detections.iterrows():
            x1 = int(detection['xmin'])
            y1 = int(detection['ymin'])
            x2 = int(detection['xmax'])
            y2 = int(detection['ymax'])
            label = f"{detection['name']} {detection['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

    def start(self):
        """Start the drone controller."""
        try:
            self.drone.connect()
            self.drone.start_stream()
            self.frame_read = self.drone.frame_read
            self.initialize_video_writer()
            self.start_time = time.time()
            self.last_action_time = self.start_time

            self.drone.takeoff()
            battery_level = self.drone.tello.get_battery()
            if battery_level < 15:
                raise DroneError(f"Battery too low after takeoff ({battery_level}%). Landing drone.")
            print(f"Drone battery level after takeoff: {battery_level}%")

            while True:
                try:
                    frame = self.drone.get_frame()
                    resized_frame = cv2.resize(frame, (640, 480))

                    current_time = time.time()

                    if current_time - self.start_time >= self.flight_duration:
                        print("Flight duration exceeded. Landing drone.")
                        break

                    if current_time - self.last_action_time >= 1.0:
                        # Perform object detection
                        target_detections = self.detector.detect(resized_frame)

                        if not target_detections.empty:
                            # Get the largest detection (assuming the closest object)
                            largest_detection = target_detections.iloc[0]

                            # Check if the object is centered before initiating orbit
                            object_x_center = (largest_detection['xmin'] + largest_detection['xmax']) / 2
                            object_y_center = (largest_detection['ymin'] + largest_detection['ymax']) / 2

                            if (
                                abs(object_x_center - self.frame_center[0])
                                <= self.object_center_tolerance
                                and abs(object_y_center - self.frame_center[1])
                                <= self.object_center_tolerance
                            ):
                                self.orbit_object()
                            else:
                                self.adjust_drone_position(largest_detection)

                        self.last_action_time = current_time

                    # Draw detections on the frame
                    target_detections = self.detector.detect(resized_frame)
                    if not target_detections.empty:
                        self.draw_detections(resized_frame, target_detections)

                    cv2.imshow('Tello Object Navigation', resized_frame)
                    self.record_frame(frame)

                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        print("User interruption. Landing drone.")
                        break

                except DroneError as e:
                    print(f"Drone error: {e}")
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break

        except DroneError as e:
            print(f"Initialization error: {e}")
        finally:
            # Cleanup
            print("Cleaning up resources.")
            self.drone.stop_stream()
            if self.out is not None:
                self.out.release()
            cv2.destroyAllWindows()
            self.drone.land()