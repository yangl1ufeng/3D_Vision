"""Live Gemini 2 color/depth preview. Press Q or Esc to close."""
import argparse
import time

import cv2
import numpy as np
from pyorbbecsdk import (AlignFilter, Config, Context, OBFormat,
                          OBFrameAggregateOutputMode, OBSensorType,
                          OBStreamType, Pipeline)


def open_camera():
    if Context().query_devices().get_count() == 0:
        raise RuntimeError('No Orbbec camera found. Check the USB connection.')
    pipeline = Pipeline()
    config = Config()
    color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    # The 1920x1080 default stalls the interactive window on this laptop.
    config.enable_stream(color.get_video_stream_profile(640, 480, OBFormat.RGB, 15))
    config.enable_stream(depth.get_video_stream_profile(640, 400, OBFormat.Y16, 15))
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.start(config)
    return pipeline, AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


def render(color, depth):
    height, width = color.get_height(), color.get_width()
    if (depth.get_height(), depth.get_width()) != (height, width):
        raise ValueError('Aligned depth and RGB dimensions differ')
    if color.get_format() != OBFormat.RGB:
        raise ValueError(f'Unexpected color format: {color.get_format()}')
    rgb = np.frombuffer(color.get_data(), np.uint8).reshape(height, width, 3)
    raw = np.frombuffer(depth.get_data(), np.uint16).reshape(height, width)
    depth_m = raw.astype(np.float32) * (float(depth.get_depth_scale()) / 1000.0)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    # Downsize before coloring to keep the interactive preview responsive.
    size = (640, 480)
    bgr_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_m, size, interpolation=cv2.INTER_NEAREST)
    valid_small = np.isfinite(depth_small) & (depth_small > 0)
    clipped = np.clip(depth_small, 0.2, 2.0)
    intensity = ((2.0 - clipped) / 1.8 * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    depth_color[~valid_small] = 0
    cv2.putText(bgr_small, 'RGB', (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv2.putText(depth_color, f'Depth 0.2-2.0 m | valid {100*valid.mean():.0f}%',
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
    return np.hstack((bgr_small, depth_color)), float(valid.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-frames', type=int, default=0,
                        help='Read N frames without opening a window; used for diagnostics')
    args = parser.parse_args()
    if args.check_frames < 0:
        parser.error('--check-frames must be nonnegative')
    pipeline, align = open_camera()
    window = 'Gemini 2 | RGB + aligned depth | Q / Esc to close'
    frames_seen = 0
    start = time.monotonic()
    try:
        if not args.check_frames:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 480)
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames:
                if time.monotonic() - start > 10 and frames_seen == 0:
                    raise RuntimeError('Camera started but no frames arrived in 10 seconds')
                if not args.check_frames:
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                continue
            aligned = align.process(frames)
            if not aligned:
                continue
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            picture, valid_fraction = render(color, depth)
            frames_seen += 1
            if args.check_frames:
                if frames_seen >= args.check_frames:
                    print(f'OK: {frames_seen} aligned RGB-D frames; valid depth {valid_fraction:.1%}')
                    break
            else:
                cv2.imshow(window, picture)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        pipeline.stop()
        if not args.check_frames:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
