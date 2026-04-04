"""Video helpers based on OpenCV.

Audio-related helpers were intentionally removed from the active pipeline to
keep inference focused on silent stereo-video generation.
"""

import cv2


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Cannot determine frame count for: {video_path}")

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
    }


def create_video_writer(video_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {video_path}")
    return writer


get_video_infos = get_video_metadata
