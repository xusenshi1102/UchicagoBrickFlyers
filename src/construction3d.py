import os
import copy
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from external.dust3r.dust3r.inference import inference
from external.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from external.dust3r.dust3r.image_pairs import make_pairs
from external.dust3r.dust3r.utils.image import load_images
from external.dust3r.dust3r.utils.device import to_numpy
from external.dust3r.dust3r.viz import (
    add_scene_cam,
    CAM_COLORS,
    OPENGL,
    pts3d_to_trimesh,
    cat_meshes
)
from external.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode


class Dust3r3DConstruction:
    """Class for 3D reconstruction using Dust3r."""

    def __init__(self, min_conf_thr, device='cuda', model_weights=None, image_size=512, silent=False):
        """
        Initialize the Dust3r3DConstruction class.

        Args:
            device (str): The device to run the model on ('cuda' or 'cpu').
            model_weights (str, optional): Path to the model weights. If None,
                default weights are used.
            image_size (int): The size to which images are resized (512 or 224).
            silent (bool): If True, suppresses output messages.
        """
        self.device = torch.device(device)
        self.image_size = image_size
        self.silent = silent
        self.min_conf_thr = 3.0

        if model_weights is None:
            # Use default model weights
            weights_path = 'external/dust3r/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
        else:
            weights_path = model_weights

        # Load the model
        self.model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True  # For GPU >= Ampere and PyTorch >= 1.12
        self.batch_size = 1

    def run_reconstruction(self, image_dir, output_dir):
        """
        Run the reconstruction process on images in `image_dir` and save the
        GLB file to `output_dir`.

        Args:
            image_dir (str): Directory containing input images.
            output_dir (str): Directory to save the output GLB file.
        """
        # Load images
        image_files = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(image_files) == 0:
            raise ValueError("No images found in the directory.")

        imgs = load_images(image_files, size=self.image_size, verbose=not self.silent)

        # Ensure there are at least two images
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1

        # Create pairs
        pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

        # Run inference
        output = inference(
            pairs,
            self.model,
            self.device,
            batch_size=self.batch_size,
            verbose=not self.silent
        )

        # Run global alignment
        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        )
        scene = global_aligner(output, device=self.device, mode=mode, verbose=not self.silent)
        lr = 0.01
        niter = 300
        schedule = 'linear'

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        # Generate 3D model
        min_conf_thr = self.min_conf_thr
        as_pointcloud = False
        mask_sky = False
        clean_depth = True
        transparent_cams = False
        cam_size = 0.05

        glb_file = self.get_3d_model_from_scene(
            output_dir, scene, min_conf_thr, as_pointcloud,
            mask_sky, clean_depth, transparent_cams, cam_size
        )

        if not self.silent:
            print(f"3D model saved to {glb_file}")

    def get_3d_model_from_scene(
        self,
        outdir,
        scene,
        min_conf_thr=3.0,
        as_pointcloud=False,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05
    ):
        """
        Extract 3D model (GLB file) from a reconstructed scene.

        Args:
            outdir (str): Output directory to save the GLB file.
            scene: Reconstructed scene object.
            min_conf_thr (float): Minimum confidence threshold.
            as_pointcloud (bool): Whether to output as point cloud.
            mask_sky (bool): Whether to mask the sky.
            clean_depth (bool): Whether to clean up depth maps.
            transparent_cams (bool): Whether to make cameras transparent.
            cam_size (float): Camera size in the output GLB file.

        Returns:
            str: Path to the saved GLB file.
        """
        if scene is None:
            return None

        # Post-processing
        if clean_depth:
            scene = scene.clean_pointcloud()
        if mask_sky:
            scene = scene.mask_sky()

        # Get optimized values from scene
        rgbimg = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()

        # 3D point cloud from depth map, poses, and intrinsics
        pts3d = to_numpy(scene.get_pts3d())
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
        mask = to_numpy(scene.get_masks())

        glb_file = self.convert_scene_output_to_glb(
            outdir, rgbimg, pts3d, mask, focals, cams2world,
            as_pointcloud, transparent_cams, cam_size
        )

        return glb_file

    def convert_scene_output_to_glb(
        self,
        outdir,
        imgs,
        pts3d,
        mask,
        focals,
        cams2world,
        as_pointcloud=False,
        transparent_cams=False,
        cam_size=0.05
    ):
        """
        Convert scene output to a GLB file.

        Args:
            outdir (str): Output directory to save the GLB file.
            imgs (list): List of images.
            pts3d (list): List of 3D points.
            mask (list): List of masks.
            focals (list): List of focal lengths.
            cams2world (list): List of camera-to-world transformations.
            as_pointcloud (bool): Whether to output as point cloud.
            transparent_cams (bool): Whether to make cameras transparent.
            cam_size (float): Camera size in the output GLB file.

        Returns:
            str: Path to the saved GLB file.
        """
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        scene = trimesh.Scene()

        # Full point cloud
        if as_pointcloud:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
            pct = trimesh.PointCloud(
                pts.reshape(-1, 3), colors=col.reshape(-1, 3)
            )
            scene.add_geometry(pct)
        else:
            meshes = []
            for i in range(len(imgs)):
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

        # Add cameras
        for i, pose_c2w in enumerate(cams2world):
            camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size
            )

        # Apply transformation to align the scene
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

        # Export the scene as GLB file
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, 'scene.glb')
        scene.export(file_obj=outfile)
        return outfile
