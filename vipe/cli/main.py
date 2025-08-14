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

from pathlib import Path

import click
import hydra
import logging
import numpy as np
from tqdm import tqdm

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream, VideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "colorization_results",
)
@click.option("--pipeline", "-p", default="colorization", help="Pipeline configuration to use (default: 'colorization')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
@click.option("--device", "-d", default="cuda", help="Device to use for processing (cuda/cpu)")
def colorize(video: Path, output: Path, pipeline: str, visualize: bool, device: str):
    """Run video colorization on a black and white video file."""

    logger = configure_logging()

    overrides = [
        f"pipeline={pipeline}", 
        f"pipeline.output.path={output}", 
        f"pipeline.colorization.device={device}",
        "pipeline.output.save_artifacts=true"
    ]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
        overrides.append("pipeline.output.save_intermediate=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Running colorization on {video}...")
    colorization_pipeline = make_pipeline(args.pipeline)

    # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
    video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")

    result = colorization_pipeline.run(video_stream)
    
    # Generate final colorized video
    if hasattr(result, 'output_streams') and result.output_streams:
        _generate_final_colorized_video(result.output_streams[0], output)
    
    logger.info("Colorization completed!")


def _generate_final_colorized_video(stream: VideoStream, output_dir: Path):
    """Generate the final colorized video from the processed stream."""
    import cv2
    
    output_video_path = output_dir / "final_colorized_video.mp4"
    logger = logging.getLogger(__name__)
    
    if len(stream) == 0:
        logger.warning("No frames to process for final video generation")
        return
    
    # Get video properties from first frame
    first_frame = next(iter(stream))
    height, width = first_frame.size()
    fps = stream.fps()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    logger.info(f"Generating final colorized video: {output_video_path}")
    
    for frame in tqdm(stream, desc="Writing final video"):
        if hasattr(frame, 'colorized_rgb') and frame.colorized_rgb is not None:
            # Convert from RGB to BGR for OpenCV
            colorized_bgr = cv2.cvtColor(frame.colorized_rgb.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(colorized_bgr)
        else:
            # Fallback to grayscale converted to BGR
            if len(frame.rgb.shape) == 2:
                gray_bgr = cv2.cvtColor(frame.rgb.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                gray_bgr = cv2.cvtColor(frame.rgb.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(gray_bgr)
    
    out.release()
    logger.info(f"Final colorized video saved: {output_video_path}")


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
def infer(video: Path, output: Path, pipeline: str, visualize: bool):
    """Run inference on a video file."""

    logger = configure_logging()

    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {video}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
    video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(colorize)
main.add_command(visualize)


if __name__ == "__main__":
    main()
