from src.controller import DroneController
from src.segmentation import SAM2Segmentation
import numpy as np
import os
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

    sam2_segmentation = SAM2Segmentation(
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path="../checkpoints/sam2.1_hiera_large.pt"
    )

    # Define SAM2 parameters
    video_path = controller.video_path
    output_dir = os.path.join('object', args.target_object, 'segmentation')
    ann_frame_idx = 2  # Frame index
    ann_obj_id = 1  # Object ID
    points = np.array([[500, 350], [480, 200]], dtype=np.float32)  # Suitcase Position
    labels = np.array([1, 1], dtype=np.int32)  # Suitcase Position 2

    # Perform segmentation
    sam2_segmentation.segment_video(
        video_path=video_path,
        output_dir=output_dir,
        ann_frame_idx=ann_frame_idx,
        ann_obj_id=ann_obj_id,
        points=points,
        labels=labels,
        frame_quality=95,
        fps=30,
        vis_frame_stride=50
    )




if __name__ == '__main__':
    main()
