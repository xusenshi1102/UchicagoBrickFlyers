import os
import numpy as np
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
from external.sam2.build_sam import build_sam2_video_predictor


class SAM2Segmentation:
    """Class for video segmentation using the SAM2 model."""

    def __init__(self, model_cfg, checkpoint_path, device=None):
        """Initialize the SAM2Segmentation class.

        Args:
            model_cfg (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint file.
            device (torch.device, optional): Device to run the model on.
                Defaults to CUDA if available.
        """
        # Set environment variables for PyTorch
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

        # Set the device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize the predictor
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint_path, device=self.device
        )

        # Initialize other attributes
        self.inference_state = None
        self.frame_names = []
        self.video_dir = ""
        self.video_segments = {}

    def extract_frames(self, video_path, output_dir, frame_quality=85, fps=1):
        """Extract frames from a video and save them as images.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the extracted frames.
            frame_quality (int, optional): Quality of the saved JPEG images.
                Defaults to 85.
            fps (int, optional): Frames per second to extract.
                Defaults to 1.
        """
        os.makedirs(output_dir, exist_ok=True)
        clip = VideoFileClip(video_path)
        frame_number = 0

        for t in np.arange(0, clip.duration, 1 / fps):
            frame = clip.get_frame(t)
            frame_image = Image.fromarray(np.uint8(frame))
            frame_path = os.path.join(output_dir, f"{frame_number:05d}.jpeg")
            frame_image.save(frame_path, quality=frame_quality)
            frame_number += 1

        self.frame_names = sorted(os.listdir(output_dir))
        self.video_dir = output_dir
        print(f"Extracted {frame_number} frames to {output_dir}")

    def initialize_inference(self):
        """Initialize the inference state for segmentation."""
        if not self.video_dir:
            raise ValueError("Video directory not set. Please extract frames first.")
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        print("Inference state initialized.")

    def add_points(self, ann_frame_idx, ann_obj_id, points, labels):
        """Add points to refine the mask for a specific frame and object.

        Args:
            ann_frame_idx (int): Frame index to interact with.
            ann_obj_id (int): Unique identifier for the object.
            points (np.ndarray): Array of point coordinates.
            labels (np.ndarray): Array of labels for the points
                (1 for positive, 0 for negative).
        """
        _, _, _ = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        print(
            f"Added points for object ID {ann_obj_id} on frame {ann_frame_idx}."
        )

    def propagate_and_save(self, output_dir, vis_frame_stride=1):
        """Propagate the segmentation masks through the video and save cropped objects.

        Args:
            output_dir (str): Directory to save the cropped object images.
            vis_frame_stride (int, optional): Frequency of frames to process.
                Defaults to 1 (process every frame).
        """
        os.makedirs(output_dir, exist_ok=True)
        torch.backends.cuda.sdp_kernel = "flash_attention"
        total_frames = len(self.frame_names)

        # Run propagation
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save cropped objects
        for out_frame_idx in range(total_frames):
            if out_frame_idx % vis_frame_stride != 0:
                continue
            frame_path = os.path.join(
                self.video_dir, self.frame_names[out_frame_idx]
            )
            frame = Image.open(frame_path).convert("RGBA")
            frame_np = np.array(frame)

            if out_frame_idx not in self.video_segments:
                continue

            for out_obj_id, out_mask in self.video_segments[
                out_frame_idx
            ].items():
                out_mask = np.squeeze(out_mask)
                alpha_channel = (out_mask * 255).astype(np.uint8)
                rgba_frame = np.dstack((frame_np[:, :, :3], alpha_channel))
                coords = np.argwhere(out_mask)
                if coords.size == 0:
                    continue
                y_min, x_min = coords[:, 0].min(), coords[:, 1].min()
                y_max, x_max = coords[:, 0].max(), coords[:, 1].max()
                cropped_object = rgba_frame[y_min:y_max, x_min:x_max]
                cropped_image = Image.fromarray(cropped_object, "RGBA")
                cropped_image_path = os.path.join(
                    output_dir,
                    f"frame{out_frame_idx:03d}_obj{out_obj_id}.png",
                )
                cropped_image.save(cropped_image_path, "PNG")
                print(
                    f"Saved cropped object {out_obj_id} from frame {out_frame_idx} to {cropped_image_path}"
                )

    def segment_video(
        self,
        video_path,
        output_dir,
        ann_frame_idx,
        ann_obj_id,
        points,
        labels,
        frame_quality=85,
        fps=30,
        vis_frame_stride=1,
    ):
        """Perform the full segmentation process on a video.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the cropped object images.
            ann_frame_idx (int): Frame index to interact with.
            ann_obj_id (int): Unique identifier for the object.
            points (np.ndarray): Array of point coordinates.
            labels (np.ndarray): Array of labels for the points
                (1 for positive, 0 for negative).
            frame_quality (int, optional): Quality of the saved JPEG images.
                Defaults to 85.
            fps (int, optional): Frames per second to extract.
                Defaults to 30.
            vis_frame_stride (int, optional): Frequency of frames to process.
                Defaults to 1 (process every frame).
        """
        self.extract_frames(
            video_path=video_path,
            output_dir=os.path.join(output_dir, "frames"),
            frame_quality=frame_quality,
            fps=fps,
        )
        self.initialize_inference()
        self.add_points(
            ann_frame_idx=ann_frame_idx,
            ann_obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        self.propagate_and_save(
            output_dir=os.path.join(output_dir, "cropped_objects"),
            vis_frame_stride=vis_frame_stride,
        )
