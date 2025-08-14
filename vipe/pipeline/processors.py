# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from typing import Iterator

import numpy as np
import torch

from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VdieoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode


logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(self.mask_phrases, sam_points_per_side=50, sam_run_gap=self.sam_run_gap)
        self.mask_expand = mask_expand

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model = VdieoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_result: torch.Tensor = unpack_optional(
            self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                focal_length = frame.intrinsics[0].item()
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(rgb=frame.rgb.float().cuda(), focal_length=focal_length)
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale, bias = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    scale, bias = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale, bias)
                scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                self.cache_scale_bias = (scale, bias)

                video_inv_depth = video_depth_inv_depth * scale + bias
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


# ============================================================================
# Colorization Processors
# ============================================================================

class SemanticSegmentationProcessor(StreamProcessor):
    """
    Processor for performing semantic segmentation on video frames.
    Provides object-level understanding for colorization.
    """
    
    def __init__(self, model_name: str = "openmmlab/upernet-convnext-small", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def _initialize_model(self):
        """Lazy initialization of the segmentation model."""
        if self.model is None:
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "image-segmentation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
                )
                logger.info(f"Initialized semantic segmentation model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize segmentation model: {e}")
                raise
    
    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.SEMANTIC_MASK}
    
    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        if self.model is None:
            self._initialize_model()
        
        try:
            # Convert frame to RGB if needed
            if len(frame.rgb.shape) == 2:  # Grayscale
                frame_rgb = torch.stack([frame.rgb, frame.rgb, frame.rgb], dim=-1)
            else:
                frame_rgb = frame.rgb
            
            # Convert to numpy for the model
            frame_np = frame_rgb.cpu().numpy().astype(np.uint8)
            
            # Perform segmentation
            segments = self.model(frame_np)
            
            # Create semantic mask
            semantic_mask = torch.zeros(frame.rgb.shape[:2], dtype=torch.uint8, device=frame.rgb.device)
            
            for i, segment in enumerate(segments):
                mask = torch.from_numpy(np.array(segment["mask"])).to(frame.rgb.device)
                semantic_mask[mask] = i + 1  # Assign class ID
            
            # Store semantic information
            frame.semantic_mask = semantic_mask
            frame.semantic_segments = segments
            
        except Exception as e:
            logger.warning(f"Semantic segmentation failed for frame {frame_idx}: {e}")
            # Create empty semantic mask as fallback
            frame.semantic_mask = torch.zeros(frame.rgb.shape[:2], dtype=torch.uint8, device=frame.rgb.device)
            frame.semantic_segments = []
        
        return frame


class PointCloudProcessor(StreamProcessor):
    """
    Processor for generating semantic point clouds from depth and semantic information.
    """
    
    def __init__(self, intrinsics: torch.Tensor, density: int = 1000):
        super().__init__()
        self.intrinsics = intrinsics
        self.density = density
        
    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.POINT_CLOUD}
    
    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        try:
            if not hasattr(frame, 'metric_depth') or not hasattr(frame, 'semantic_mask'):
                logger.warning(f"Missing depth or semantic data for frame {frame_idx}")
                return frame
            
            # Get camera intrinsics
            fx, fy, cx, cy = self.intrinsics[:4]
            
            # Get depth and semantic data
            depth_map = frame.metric_depth.cpu().numpy()
            semantic_mask = frame.semantic_mask.cpu().numpy()
            
            h, w = depth_map.shape
            
            # Sample points based on density parameter
            total_pixels = h * w
            step = max(1, int(np.sqrt(total_pixels / self.density)))
            
            points_3d = []
            semantics = []
            colors = []
            
            for v in range(0, h, step):
                for u in range(0, w, step):
                    z = depth_map[v, u]
                    if z > 0:  # Valid depth
                        # Unproject to 3D
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        
                        points_3d.append([x, y, z])
                        semantics.append(semantic_mask[v, u])
                        
                        # Get color from frame (grayscale to RGB)
                        if len(frame.rgb.shape) == 2:
                            gray_val = frame.rgb[v, u].item()
                            colors.append([gray_val, gray_val, gray_val])
                        else:
                            colors.append(frame.rgb[v, u].cpu().numpy())
            
            if points_3d:
                frame.point_cloud = {
                    'points': np.array(points_3d),
                    'semantics': np.array(semantics),
                    'colors': np.array(colors)
                }
            else:
                frame.point_cloud = None
                
        except Exception as e:
            logger.warning(f"Point cloud generation failed for frame {frame_idx}: {e}")
            frame.point_cloud = None
        
        return frame


class ColorizationProcessor(StreamProcessor):
    """
    Main colorization processor that uses depth and semantic information
    to generate temporally consistent and semantically plausible colors.
    """
    
    def __init__(
        self, 
        model_name: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda",
        inference_steps: int = 20,
        guidance_scale: float = 7.5,
        temporal_weight: float = 0.3,
        use_depth_conditioning: bool = True,
        use_semantic_conditioning: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.temporal_weight = temporal_weight
        self.use_depth_conditioning = use_depth_conditioning
        self.use_semantic_conditioning = use_semantic_conditioning
        
        self.model = None
        self.previous_colorized = None
        
    def _initialize_model(self):
        """Lazy initialization of the colorization model."""
        if self.model is None:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                self.model = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                logger.info(f"Initialized colorization model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize colorization model: {e}")
                raise
    
    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.COLORIZED_RGB}
    
    def _create_conditioning_prompt(self, frame: VideoFrame) -> str:
        """Create a conditioning prompt based on semantic and depth information."""
        base_prompt = "a realistic colorized photograph"
        
        if self.use_semantic_conditioning and hasattr(frame, 'semantic_segments'):
            # Extract dominant objects from semantic segmentation
            objects = []
            for segment in frame.semantic_segments[:3]:  # Top 3 segments
                if 'label' in segment:
                    objects.append(segment['label'])
            
            if objects:
                objects_str = ", ".join(objects)
                base_prompt = f"a realistic colorized photograph with {objects_str}"
        
        return base_prompt
    
    def _apply_temporal_consistency(self, current_frame: torch.Tensor, 
                                  previous_frame: torch.Tensor) -> torch.Tensor:
        """Apply temporal consistency between consecutive frames."""
        if self.previous_colorized is None:
            return current_frame
        
        # Simple temporal smoothing
        alpha = self.temporal_weight
        return alpha * previous_frame + (1 - alpha) * current_frame
    
    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        if self.model is None:
            self._initialize_model()
        
        try:
            # Convert grayscale to RGB if needed
            if len(frame.rgb.shape) == 2:
                gray_rgb = torch.stack([frame.rgb, frame.rgb, frame.rgb], dim=-1)
            else:
                gray_rgb = frame.rgb
            
            # Convert to PIL Image for diffusion model
            from PIL import Image
            gray_np = gray_rgb.cpu().numpy().astype(np.uint8)
            gray_pil = Image.fromarray(gray_np)
            
            # Create mask for full image colorization
            mask = np.ones(gray_rgb.shape[:2], dtype=np.uint8) * 255
            mask_pil = Image.fromarray(mask)
            
            # Create conditioning prompt
            prompt = self._create_conditioning_prompt(frame)
            
            # Generate colorized image
            colorized = self.model(
                prompt=prompt,
                image=gray_pil,
                mask_image=mask_pil,
                height=gray_rgb.shape[0],
                width=gray_rgb.shape[1],
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale
            ).images[0]
            
            # Convert back to tensor
            colorized_np = np.array(colorized)
            colorized_tensor = torch.from_numpy(colorized_np).to(frame.rgb.device)
            
            # Apply temporal consistency if we have a previous frame
            if self.previous_colorized is not None:
                colorized_tensor = self._apply_temporal_consistency(
                    colorized_tensor, self.previous_colorized
                )
            
            # Store colorized result
            frame.colorized_rgb = colorized_tensor
            self.previous_colorized = colorized_tensor.clone()
            
        except Exception as e:
            logger.warning(f"Colorization failed for frame {frame_idx}: {e}")
            # Fallback: convert grayscale to RGB
            if len(frame.rgb.shape) == 2:
                frame.colorized_rgb = torch.stack([frame.rgb, frame.rgb, frame.rgb], dim=-1)
            else:
                frame.colorized_rgb = frame.rgb.clone()
        
        return frame
