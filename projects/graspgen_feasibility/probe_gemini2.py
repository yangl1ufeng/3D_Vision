"""Capture one aligned Gemini 2 RGB-D pair and report camera calibration.

This probe does not assert a tabletop/world calibration or a target mask.
"""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pyorbbecsdk import (AlignFilter, Config, Context, OBFormat,
                          OBFrameAggregateOutputMode, OBSensorType,
                          OBStreamType, Pipeline)

ROOT = Path(__file__).resolve().parent


def parameters(value):
    return {name: getattr(value, name) for name in ('fx', 'fy', 'cx', 'cy', 'width', 'height')
            if hasattr(value, name)}


def distortion(value):
    return {name: str(getattr(value, name)) if name == 'model' else float(getattr(value, name))
            for name in ('model', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2')
            if hasattr(value, name)}


def main():
    devices = Context().query_devices()
    if devices.get_count() == 0:
        raise RuntimeError('Orbbec SDK found no camera')
    pipeline = Pipeline()
    config = Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pipeline.start(config)
    try:
        deadline = time.monotonic() + 25
        best = None
        frame_count = 0
        for _ in range(60):
            if time.monotonic() > deadline:
                break
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            aligned = align.process(frames)
            if not aligned:
                continue
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            h, w = color.get_height(), color.get_width()
            if (depth.get_height(), depth.get_width()) != (h, w):
                continue
            rgb = np.frombuffer(color.get_data(), np.uint8).reshape(h, w, 3).copy()
            raw = np.frombuffer(depth.get_data(), np.uint16).reshape(h, w)
            # Orbbec SDK reports a multiplier to millimetres (official example).
            scale_mm = float(depth.get_depth_scale())
            depth_m = (raw.astype(np.float32) * (scale_mm / 1000.0))
            valid = np.isfinite(depth_m) & (depth_m > 0)
            if valid.sum() < 1000:
                continue
            frame_count += 1
            color_intrinsic = color.get_stream_profile().get_intrinsic()
            depth_intrinsic = depth.get_stream_profile().get_intrinsic()
            sample = (rgb, depth_m, valid, color_intrinsic, depth_intrinsic,
                      color.get_stream_profile().get_distortion(),
                      depth.get_stream_profile().get_distortion(),
                      scale_mm, str(color.get_format()), str(depth.get_format()))
            if best is None or valid.sum() > best[2].sum():
                best = sample
        if best is None:
            raise RuntimeError('No complete aligned RGB-D frame within 25 seconds')
        (rgb, depth_m, valid, color_intrinsic, depth_intrinsic,
         color_distortion, depth_distortion, scale_mm, color_format, depth_format) = best
        h, w = rgb.shape[:2]
        out = ROOT/'runs/gemini2_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True)
        Image.fromarray(rgb).save(out/'rgb.png')
        np.save(out/'depth_m.npy', depth_m)
        np.save(out/'valid_mask.npy', valid.astype(np.uint8))
        report = {'device_count': devices.get_count(), 'frames_checked': frame_count,
                  'rgb_shape': [h,w,3],
                  'depth_shape': [h,w], 'depth_scale_mm_per_unit': scale_mm,
                  'valid_pixels': int(valid.sum()), 'valid_fraction': float(valid.mean()),
                  'saturated_rgb_fraction': float((rgb.min(axis=2) >= 250).mean()),
                  'depth_range_m_p05_p50_p95': np.percentile(depth_m[valid], [5,50,95]).tolist(),
                  'color_intrinsic': parameters(color_intrinsic),
                  'aligned_depth_intrinsic': parameters(depth_intrinsic),
                  'color_distortion': distortion(color_distortion),
                  'aligned_depth_distortion': distortion(depth_distortion),
                  'color_format': color_format,
                  'depth_format': depth_format,
                  'note': 'Aligned D2C frame. Camera-to-table transform and target mask not yet measured.'}
        (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        print(out, flush=True)
        return
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
