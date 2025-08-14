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

"""
Video Colorization Pipeline Integration with ViPE

This module provides a ViPE-integrated pipeline for AI-based video colorization
that leverages ViPE's robust 3D geometric perception capabilities.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from vipe.pipeline import Pipeline, AnnotationPipelineOutput
from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

# Import the colorization processors
try:
    from .processors import ColorizationProcessor, SemanticSegmentationProcessor, PointCloudProcessor
except ImportError:
    # Fallback imports in case of missing dependencies
    ColorizationProcessor = None
    SemanticSegmentationProcessor = None
    PointCloudProcessor = None


logger = logging.getLogger(__name__)


class VideoColorizationPipeline(Pipeline):
    """
    ViPE-integrated video colorization pipeline.
    
    This pipeline extends ViPE's default annotation pipeline to include
    colorization capabilities while maintaining all of ViPE's robust
    3D geometric perception features.
    """
    
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, 
                 colorization: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.colorization_cfg = colorization
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)
        
        # Initialize colorization-specific paths
        self.colorization_path = self.out_path / "colorization"
        self.colorization_path.mkdir(exist_ok=True, parents=True)
        
    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        """Add initialization processors including ViPE's standard processors."""
        init_processors: list[StreamProcessor] = []

        # Standard ViPE processors
        from .processors import GeoCalibIntrinsicsProcessor, TrackAnythingProcessor
        
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        """Add post-processing including depth alignment and colorization processors."""
        from .processors import AdaptiveDepthProcessor
        
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        
        # Add depth alignment processor
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
            
        return ProcessedVideoStream(video_stream, post_processors)
    
    def _add_colorization_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        """Add colorization-specific processors."""
        colorization_processors: list[StreamProcessor] = []
        
        # Check if colorization processors are available
        if SemanticSegmentationProcessor is None:
            logger.warning("Semantic segmentation processor not available. Skipping semantic processing.")
        else:
            # Add semantic segmentation processor
            colorization_processors.append(
                SemanticSegmentationProcessor(
                    model_name=self.colorization_cfg.semantic_model,
                    device=self.colorization_cfg.device
                )
            )
        
        # Add point cloud generation processor
        if PointCloudProcessor is not None and self.colorization_cfg.get("save_point_clouds", True):
            colorization_processors.append(
                PointCloudProcessor(
                    intrinsics=slam_output.intrinsics[view_idx],
                    density=self.colorization_cfg.get("point_cloud_density", 1000)
                )
            )
        
        # Add main colorization processor
        if ColorizationProcessor is None:
            logger.warning("Colorization processor not available. Skipping colorization processing.")
        else:
            colorization_processors.append(
                ColorizationProcessor(
                    model_name=self.colorization_cfg.colorization_model,
                    device=self.colorization_cfg.device,
                    inference_steps=self.colorization_cfg.get("inference_steps", 20),
                    guidance_scale=self.colorization_cfg.get("guidance_scale", 7.5),
                    temporal_weight=self.colorization_cfg.get("temporal_weight", 0.3),
                    use_depth_conditioning=self.colorization_cfg.get("depth_conditioning", True),
                    use_semantic_conditioning=self.colorization_cfg.get("semantic_consistency", True)
                )
            )
        
        return ProcessedVideoStream(video_stream, colorization_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        """Run the complete colorization pipeline."""
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()
        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been processed already, skip it!!")
            return annotate_output

        # Phase 1: ViPE SLAM processing for robust 3D perception
        logger.info("Phase 1: Running ViPE SLAM for 3D geometric perception...")
        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        # Phase 2: Post-processing for depth alignment
        logger.info("Phase 2: Post-processing for temporally consistent depth...")
        post_processed_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Phase 3: Colorization processing
        logger.info("Phase 3: Running colorization pipeline...")
        colorized_streams = [
            self._add_colorization_processors(view_idx, post_stream, slam_output).cache("colorization", online=True)
            for view_idx, post_stream in enumerate(post_processed_streams)
        ]

        # Phase 4: Save outputs and generate artifacts
        logger.info("Phase 4: Saving outputs and generating artifacts...")
        for colorized_stream, artifact_path in zip(colorized_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, colorized_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            # Save colorization-specific artifacts
            if self.out_cfg.get("save_intermediate", True):
                self._save_colorization_artifacts(colorized_stream, artifact_path)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    colorized_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )
                
                # Save colorization visualization
                self._save_colorization_visualization(colorized_stream, artifact_path)

        if self.return_output_streams:
            annotate_output.output_streams = colorized_streams

        logger.info("Video colorization pipeline completed successfully!")
        return annotate_output
    
    def _save_colorization_artifacts(self, stream: VideoStream, artifact_path: io.ArtifactPath):
        """Save colorization-specific artifacts."""
        colorization_artifacts_path = artifact_path.base_path / "colorization"
        colorization_artifacts_path.mkdir(exist_ok=True, parents=True)
        
        # Save individual colorized frames if requested
        if self.out_cfg.get("save_colorized_frames", True):
            frames_path = colorization_artifacts_path / "frames"
            frames_path.mkdir(exist_ok=True, parents=True)
            
            for frame_idx, frame in enumerate(stream):
                if hasattr(frame, 'colorized_rgb'):
                    frame_path = frames_path / f"colorized_{frame_idx:06d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame.colorized_rgb, cv2.COLOR_RGB2BGR))
        
        # Save depth maps if requested
        if self.out_cfg.get("save_depth_maps", True):
            depth_path = colorization_artifacts_path / "depth"
            depth_path.mkdir(exist_ok=True, parents=True)
            
            for frame_idx, frame in enumerate(stream):
                if hasattr(frame, 'depth'):
                    depth_file = depth_path / f"depth_{frame_idx:06d}.npy"
                    np.save(depth_file, frame.depth)
        
        # Save semantic masks if requested
        if self.out_cfg.get("save_semantic_masks", True):
            semantic_path = colorization_artifacts_path / "semantic"
            semantic_path.mkdir(exist_ok=True, parents=True)
            
            for frame_idx, frame in enumerate(stream):
                if hasattr(frame, 'semantic_mask'):
                    semantic_file = semantic_path / f"semantic_{frame_idx:06d}.npy"
                    np.save(semantic_file, frame.semantic_mask)
    
    def _save_colorization_visualization(self, stream: VideoStream, artifact_path: io.ArtifactPath):
        """Save colorization-specific visualization videos."""
        colorization_vis_path = artifact_path.base_path / "colorization_vis"
        colorization_vis_path.mkdir(exist_ok=True, parents=True)
        
        # Create side-by-side comparison video
        comparison_video_path = colorization_vis_path / "before_after_comparison.mp4"
        
        if len(stream) > 0:
            first_frame = next(iter(stream))
            height, width = first_frame.rgb.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(comparison_video_path), fourcc, stream.fps(), (width * 2, height))
            
            for frame in stream:
                if hasattr(frame, 'colorized_rgb'):
                    # Create side-by-side comparison
                    gray_rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_GRAY2RGB) if len(frame.rgb.shape) == 2 else frame.rgb
                    colorized_rgb = frame.colorized_rgb
                    
                    # Resize if necessary
                    if gray_rgb.shape[:2] != (height, width):
                        gray_rgb = cv2.resize(gray_rgb, (width, height))
                    if colorized_rgb.shape[:2] != (height, width):
                        colorized_rgb = cv2.resize(colorized_rgb, (width, height))
                    
                    comparison = np.hstack([gray_rgb, colorized_rgb])
                    out.write(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            out.release()
            logger.info(f"Saved colorization comparison video: {comparison_video_path}")