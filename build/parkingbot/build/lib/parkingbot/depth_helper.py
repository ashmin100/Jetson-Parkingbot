"""
RealSense depth 간단 헬퍼
- pyrealsense2 파이프라인 초기화
- color/depth 프레임 수집, 특정 픽셀 거리(m) 조회
"""

from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

# Depth 스트림 설정 (poseNet RGB 해상도와 맞춤)
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30


class DepthCamera:
    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
        # color 정렬을 위해 color 스트림도 활성화
        cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)

        self.profile = self.pipe.start(cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # raw → meter 변환 계수
        self.align = rs.align(rs.stream.color)  # depth를 color 좌표계로 정렬
        self.running = True

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """정렬된 color/depth 프레임을 numpy array로 반환 (없으면 (None, None))"""
        if not self.running:
            return None, None
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_np = np.asanyarray(depth_frame.get_data())
        color_np = np.asanyarray(color_frame.get_data())  # BGR8
        return color_np, depth_np

    def get_distance(self, depth_image: np.ndarray, x: float, y: float) -> Optional[float]:
        """(x, y) 픽셀 좌표의 거리(m) 반환"""
        if depth_image is None:
            return None
        h, w = depth_image.shape[:2]
        xx = int(np.clip(x, 0, w - 1))
        yy = int(np.clip(y, 0, h - 1))
        depth_raw = float(depth_image[yy, xx])
        return depth_raw * self.depth_scale

    def stop(self):
        """RealSense 파이프라인 정지"""
        if self.running:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.running = False
