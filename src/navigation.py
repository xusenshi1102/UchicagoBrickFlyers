import time
from djitellopy import Tello


class DroneError(Exception):
    """Custom exception class for drone errors."""
    pass


class TelloDrone:
    """Class to manage Tello drone operations."""

    def __init__(self, drone_ip):
        """Initialize the Tello drone."""
        self.tello = Tello(drone_ip)
        self.speed = 10
        self.frame_read = None

    def connect(self):
        """Connect to the drone and set speed.

        Raises:
            DroneError: If the battery level is below 20%.
        """
        self.tello.connect()
        battery_level = self.tello.get_battery()
        if battery_level < 20:
            raise DroneError(f"Battery too low ({battery_level}%). Cannot proceed.")
        print(f"Drone battery level: {battery_level}%")
        self.tello.set_speed(self.speed)

    def start_stream(self):
        """Start the video stream from the drone."""
        self.tello.streamon()
        time.sleep(2)  # Allow stream to initialize
        self.frame_read = self.tello.get_frame_read()

    def get_frame(self):
        """Get the current frame from the video stream.

        Returns:
            numpy.ndarray: The current video frame.

        Raises:
            DroneError: If the frame is not available.
        """
        if self.frame_read and self.frame_read.frame is not None:
            return self.frame_read.frame
        else:
            raise DroneError("Failed to read frame from drone.")

    def takeoff(self):
        """Command the drone to take off."""
        self.tello.takeoff()

    def land(self):
        """Command the drone to land."""
        self.tello.land()

    def move_up(self, distance):
        """Move the drone up by a certain distance."""
        self.tello.move_up(distance)

    def move_down(self, distance):
        """Move the drone down by a certain distance."""
        self.tello.move_down(distance)

    def move_left(self, distance):
        """Move the drone left by a certain distance."""
        self.tello.move_left(distance)

    def move_right(self, distance):
        """Move the drone right by a certain distance."""
        self.tello.move_right(distance)

    def move_forward(self, distance):
        """Move the drone forward by a certain distance."""
        self.tello.move_forward(distance)

    def move_back(self, distance):
        """Move the drone back by a certain distance."""
        self.tello.move_back(distance)

    def rotate_clockwise(self, degrees):
        """Rotate the drone clockwise by a certain angle."""
        self.tello.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees):
        """Rotate the drone counter-clockwise by a certain angle."""
        self.tello.rotate_counter_clockwise(degrees)

    def stop_stream(self):
        """Stop the video stream from the drone."""
        self.tello.streamoff()
