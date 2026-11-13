"""Capture a frozen Gemini 2 desk layout by fusing aligned depth frames."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pyorbbecsdk import (AlignFilter, Config, Context, OBFormat,
                          OBFrameAggregateOutputMode, OBSensorType,
                          OBStreamType, Pipeline)

from offline_core.static_depth import fuse_static_depth
from probe_gemini2 import parameters, distortion

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=8)
    args = parser.parse_args()
    if not 5 <= args.frames <= 60 or not 0 <= args.warmup <= 60:
        parser.error('frames must be 5..60 and warmup 0..60')
    if Context().query_devices().get_count() == 0:
        raise RuntimeError('Orbbec SDK found no camera')
    pipeline = Pipeline()
    config = Config()
    color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(color.get_video_stream_profile(1280, 720, OBFormat.RGB, 15))
    config.enable_stream(depth.get_video_stream_profile(1280, 800, OBFormat.Y16, 15))
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pipeline.start(config)
    samples = []
    rgb = None
    start = time.monotonic()
    try:
        while len(samples) < args.frames:
            if time.monotonic() - start > 90:
                raise RuntimeError(f'Only {len(samples)} complete aligned frames arrived within 90 seconds')
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            aligned = align.process(frames)
            if not aligned:
                continue
            cf, df = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df or cf.get_format() != OBFormat.RGB:
                continue
            h, w = cf.get_height(), cf.get_width()
            if (df.get_height(), df.get_width()) != (h, w):
                continue
            if args.warmup:
                args.warmup -= 1
                continue
            image = np.frombuffer(cf.get_data(), np.uint8).reshape(h, w, 3).copy()
            raw = np.frombuffer(df.get_data(), np.uint16).reshape(h, w)
            depth_m = raw.astype(np.float32) * (float(df.get_depth_scale()) / 1000.)
            samples.append(depth_m)
            rgb = image
            if len(samples) == 1 or len(samples) % 5 == 0:
                print(f'Captured {len(samples)}/{args.frames} aligned frames', flush=True)
            color_intrinsic = cf.get_stream_profile().get_intrinsic()
            aligned_depth_intrinsic = df.get_stream_profile().get_intrinsic()
            color_distortion = cf.get_stream_profile().get_distortion()
            aligned_depth_distortion = df.get_stream_profile().get_distortion()
    finally:
        pipeline.stop()
    capture_seconds = time.monotonic()-start
    fused, counts, quality = fuse_static_depth(samples, min_observations=max(3, args.frames // 3))
    if quality['valid_fraction'] < .1:
        raise RuntimeError(f'Fused depth coverage too low: {quality["valid_fraction"]:.1%}')
    out = ROOT/'runs/gemini_static'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    Image.fromarray(rgb).save(out/'rgb.png')
    np.save(out/'depth_m.npy', fused)
    np.save(out/'valid_count.npy', counts)
    np.save(out/'valid_mask.npy', (fused > 0).astype(np.uint8))
    report = {'source': 'Orbbec Gemini 2, repeated fixed-camera D2C aligned frames',
              'color_shape': list(rgb.shape), 'depth_shape': list(fused.shape),
              'depth_unit': 'm', 'frames_requested': args.frames,
              'capture_seconds': capture_seconds,
              'quality': quality,
              'color_intrinsic': parameters(color_intrinsic),
              'aligned_depth_intrinsic': parameters(aligned_depth_intrinsic),
              'color_distortion': distortion(color_distortion),
              'aligned_depth_distortion': distortion(aligned_depth_distortion),
              'camera_pose_assumption': 'Camera fixed and desk layout unchanged during burst',
              'world_calibration_available': False,
              'target_segmentation_available': False}
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    main()
