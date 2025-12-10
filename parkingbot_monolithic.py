#!/usr/bin/env python3
"""
ROS2 노드형 parkingbot
- poseNet(resnet18-body)로 제스처 인식
- RealSense depth로 거리 필터
- 인식된 제스처를 TurtleBot3 /cmd_vel에 매핑
- jetson_utils videoOutput으로 원격 스트림 옵션 지원 (web_stream_posenet.py 참고)
"""

import math
import time
from typing import Dict, Optional

import numpy as np

# --- Jetson Inference (RGB pose) ---
from jetson_inference import poseNet
from jetson_utils import cudaToNumpy, videoOutput, videoSource

# --- RealSense (Depth) ---
import pyrealsense2 as rs

# --- ROS2 ---
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


# ==============================
# 하드코딩 가능한 튜닝 상수
# ==============================
MODEL_NAME = "resnet18-body"
VIDEO_DEVICE = "/dev/video4"  # RealSense RGB가 잡히는 비디오 디바이스
VIDEO_OUTPUT = None  # 예: "rtp://<ip>:1234", None이면 스트림 출력 안 함
LOOP_DT = 0.1  # 10Hz

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30

MIN_DIST = 0.8   # 유효 거리 최소(m)
MAX_DIST = 3.0   # 유효 거리 최대(m)

V_FORWARD = 0.15    # m/s
V_BACKWARD = -0.10  # m/s
W_TURN = 0.30       # rad/s

Y_TOL_RATIO = 0.06  # 어깨-손목 높이 허용 오차 비율
X_SIDE_RATIO = 0.15 # 어깨-손목 가로 간격
Y_DOWN_RATIO = 0.05 # 팔 내려간 판단 여유
Y_UP_RATIO = 0.10   # 팔 올린 판단 여유


# ==============================
# 제스처 상태 정의
# ==============================
class GestureState:
    STOP = "STOP"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    UNKNOWN = "UNKNOWN"


# ==============================
# RealSense Depth 래퍼
# ==============================
class DepthCamera:
    """pyrealsense2 최소 래퍼 (색상 프레임 정렬 포함)"""

    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
        # 컬러 정렬용으로 color 스트림도 켜두면 depth가 color 좌표계로 align 가능
        cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)

        self.profile = self.pipe.start(cfg)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def get_depth_image(self) -> Optional[np.ndarray]:
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            return None
        return np.asanyarray(depth_frame.get_data())

    def get_distance_m(self, depth_image: np.ndarray, x: float, y: float) -> Optional[float]:
        """poseNet 픽셀 좌표에서 깊이(m)를 리턴"""
        if depth_image is None:
            return None
        h, w = depth_image.shape[:2]
        xx = int(np.clip(x, 0, w - 1))
        yy = int(np.clip(y, 0, h - 1))
        depth_raw = float(depth_image[yy, xx])
        return depth_raw * self.depth_scale


# ==============================
# 제스처 판단 함수
# ==============================
def detect_gesture(pose, img_width: int, img_height: int) -> str:
    """poseNet pose 객체 하나를 받아 GestureState 반환"""
    # BodyPart enum이 없는 버전 호환: COCO ID 기준으로 직접 조회 (5/6 어깨, 9/10 손목)
    kp: Dict[int, object] = {k.ID: k for k in pose.Keypoints}
    try:
        L_sh, R_sh = kp[5], kp[6]
        L_wr, R_wr = kp[9], kp[10]
    except KeyError:
        return GestureState.UNKNOWN

    L_sh_x, L_sh_y = L_sh.x, L_sh.y
    R_sh_x, R_sh_y = R_sh.x, R_sh.y
    L_wr_x, L_wr_y = L_wr.x, L_wr.y
    R_wr_x, R_wr_y = R_wr.x, R_wr.y

    y_tol = img_height * Y_TOL_RATIO
    x_side = img_width * X_SIDE_RATIO
    y_down = img_height * Y_DOWN_RATIO
    y_up = img_height * Y_UP_RATIO

    # --- 1) 차렷: STOP ---
    arms_down = (
        L_wr_y > L_sh_y + y_down and
        R_wr_y > R_sh_y + y_down and
        abs(L_wr_x - L_sh_x) < x_side and
        abs(R_wr_x - R_sh_x) < x_side
    )
    if arms_down:
        return GestureState.STOP

    # --- 2) 양팔 수평: FORWARD ---
    left_side = abs(L_wr_y - L_sh_y) < y_tol and L_wr_x < L_sh_x - x_side
    right_side = abs(R_wr_y - R_sh_y) < y_tol and R_wr_x > R_sh_x + x_side
    if left_side and right_side:
        return GestureState.FORWARD

    # --- 3) 오른팔만 옆으로: TURN_RIGHT ---
    if right_side and not left_side and L_wr_y > L_sh_y + y_down:
        return GestureState.TURN_RIGHT

    # --- 4) 왼팔만 옆으로: TURN_LEFT ---
    if left_side and not right_side and R_wr_y > R_sh_y + y_down:
        return GestureState.TURN_LEFT

    # --- 5) 양팔 머리 위: BACKWARD ---
    if L_wr_y < L_sh_y - y_up and R_wr_y < R_sh_y - y_up:
        return GestureState.BACKWARD

    return GestureState.UNKNOWN


# ==============================
# ROS2 노드: ParkingBot
# ==============================
class ParkingBotNode(Node):
    def __init__(self):
        super().__init__("parkingbot")

        # ROS2 publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Jetson 입력/출력
        self.video_in = videoSource(
            VIDEO_DEVICE,
            argv=[
                "--input-width=640",
                "--input-height=480",
                "--input-rate=30",
            ],
        )
        self.video_out = videoOutput(VIDEO_OUTPUT) if VIDEO_OUTPUT else None

        # poseNet 초기화
        self.pose_net = poseNet(MODEL_NAME)

        # RealSense depth
        self.depth_cam = DepthCamera()

        # 타이머 (10Hz)
        self.timer = self.create_timer(LOOP_DT, self.loop_once)

    def loop_once(self):
        # 1) 컬러 프레임 캡처
        img_cuda = self.video_in.Capture()
        if img_cuda is None:
            self.get_logger().warn("No RGB frame. STOP")
            self._publish_stop()
            return

        frame = cudaToNumpy(img_cuda)
        img_h, img_w = frame.shape[:2]

        # 2) 포즈 추정 (overlay 켜면 스트림으로 바로 볼 수 있음)
        poses = self.pose_net.Process(img_cuda, overlay="links,keypoints")
        if len(poses) == 0:
            self._publish_stop()
            self._maybe_stream(img_cuda, "No person")
            return

        pose = poses[0]  # 가장 자신도 높은 첫 번째만 사용

        # 3) Depth 프레임 + 거리 계산
        depth_image = self.depth_cam.get_depth_image()
        if depth_image is None:
            self.get_logger().warn("No depth frame. STOP")
            self._publish_stop()
            return

        center_x, center_y = pose.Center[0], pose.Center[1]
        dist = self.depth_cam.get_distance_m(depth_image, center_x, center_y)
        if dist is None:
            self._publish_stop()
            return

        if dist < MIN_DIST or dist > MAX_DIST:
            self.get_logger().info(f"Distance {dist:.2f}m out of range -> STOP")
            self._publish_stop()
            self._maybe_stream(img_cuda, f"dist={dist:.2f} STOP")
            return

        # 4) 제스처 인식
        gesture = detect_gesture(pose, img_w, img_h)
        self.get_logger().info(f"Gesture={gesture} dist={dist:.2f}m")

        # 5) 제스처 -> Twist 매핑
        twist = Twist()
        if gesture == GestureState.FORWARD:
            twist.linear.x = V_FORWARD
        elif gesture == GestureState.BACKWARD:
            twist.linear.x = V_BACKWARD
        elif gesture == GestureState.TURN_LEFT:
            twist.angular.z = +W_TURN
        elif gesture == GestureState.TURN_RIGHT:
            twist.angular.z = -W_TURN
        # UNKNOWN/STOP는 기본값 0,0

        self.cmd_pub.publish(twist)
        self._maybe_stream(img_cuda, f"{gesture} {dist:.2f}m")

    def _publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def _maybe_stream(self, img_cuda, status: str = ""):
        """videoOutput이 설정된 경우 스트림 송출"""
        if not self.video_out:
            return
        self.video_out.Render(img_cuda)
        if status:
            self.video_out.SetStatus(status)


def main():
    # argparse로 스트림 출력 URI를 덮어쓸 수 있게 함
    import argparse

    parser = argparse.ArgumentParser(description="parkingbot gesture -> cmd_vel")
    parser.add_argument(
        "--output",
        type=str,
        default=VIDEO_OUTPUT,
        help="videoOutput URI (예: rtp://<ip>:1234, webrtc://@:8554, rtsp://@:8554/live)",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    # VIDEO_OUTPUT 기본값 또는 --output 인자 사용
    global VIDEO_OUTPUT
    VIDEO_OUTPUT = args.output
    node = ParkingBotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
