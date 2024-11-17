from src.controller import DroneController
import argparse

def main():
    """Main function to start the drone controller."""
    parser = argparse.ArgumentParser(description='Drone Object Tracking')
    parser.add_argument(
        '--target_object',
        type=str,
        required=True,
        help='Name of the target object to track.'
    )
    parser.add_argument(
        '--drone_ip',
        type=str,
        required=True,
        help='IP address of the drone.'
    )
    args = parser.parse_args()

    controller = DroneController(target_object=args.target_object, drone_ip=args.drone_ip)
    controller.start()

if __name__ == '__main__':
    main()
