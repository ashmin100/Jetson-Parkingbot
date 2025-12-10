#!/usr/bin/env python3
"""
ROS2 parkingbot 노드
- poseNet(resnet18-body) + RealSense depth로 제스처 인식
- 제스처를 TurtleBot3 /cmd_vel Twist로 매핑
- 선택적으로 jetson_utils videoOutput으로 스트리밍 (web_stream_posenet.py 참고)
"""

import argparse
from typing import Optional, Tuple

from geometry_msgs.msg import Twist
from jetson_utils import cudaFromNumpy, videoOutput
import rclpy
from rclpy.node import Node

from parkingbot.depth_helper import DepthCamera
from parkingbot.gesture_logic import GestureState, detect_gesture
from parkingbot.pose_wrapper import PoseEstimator

# ==============================
# 하드코딩 가능한 튜닝 상수
# ==============================
LOOP_DT = 0.1                  # 10Hz 주기

# 거리 필터: 너무 가까운 경우도 허용하도록 MIN_DIST를 0.0으로 설정
MIN_DIST = 0.0   # 제스처 인식 유효 거리 최소(m)
MAX_DIST = 3.0   # 제스처 인식 유효 거리 최대(m)

V_FORWARD = 0.12    # m/s
V_BACKWARD = -0.08  # m/s
W_TURN = 0.25       # rad/s


class ParkingBotNode(Node):
    def __init__(self, output_uri: Optional[str] = None, enable_motor_power: bool = False):
        super().__init__("parkingbot")

        # /cmd_vel publisher 준비
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # RealSense depth/color 래퍼 (pyrealsense2)
        self.depth_cam = DepthCamera()
        # 스트리밍 출력 (옵션)
        self.video_out = videoOutput(output_uri) if output_uri else None

        # poseNet 래퍼
        self.pose_estimator = PoseEstimator()

        # 모터 파워를 켜는 서비스 호출 옵션
        if enable_motor_power:
            self._try_enable_motor_power()

        # 주기적 타이머 (10Hz)
        self.timer = self.create_timer(LOOP_DT, self.loop_once)

    def loop_once(self):
        # 1) RGB/Depth 프레임 캡처 (pyrealsense2 → numpy)
        color_np, depth_image = self.depth_cam.get_frames()
        if color_np is None or depth_image is None:
            self.get_logger().warn("No RGB frame. STOP")
            self._publish_stop()
            return

        img_h, img_w = color_np.shape[:2]
        img_cuda = cudaFromNumpy(color_np)  # jetson_utils CUDA 이미지로 변환

        # 2) poseNet 추정 (overlay 켜면 스트림으로 확인 가능)
        poses = self.pose_estimator.estimate(img_cuda, overlay="links,keypoints")
        if len(poses) == 0:
            self._publish_stop()
            self._maybe_stream(img_cuda, "No person")
            return

        # 여러 사람이 검출되면 화면에서 가장 큰(영역이 넓은) 포즈를 사용
        pose = max(poses, key=self._pose_area)

        # 3) Depth 거리 계산 (포즈 중심: keypoints 평균)
        center = self._get_pose_center(pose)
        if center is None:
            self._publish_stop()
            self._maybe_stream(img_cuda, "No center")
            return
        center_x, center_y = center
        dist = self.depth_cam.get_distance(depth_image, center_x, center_y)
        if dist is None:
            self._publish_stop()
            return

        # 거리 필터: 사람이 너무 멀거나 가까우면 안전 정지
        if dist < MIN_DIST or dist > MAX_DIST:
            self.get_logger().info(f"Distance {dist:.2f}m out of range -> STOP")
            self._publish_stop()
            self._maybe_stream(img_cuda, f"dist={dist:.2f} STOP")
            return

        # 4) 제스처 판별 (어깨/손목 좌표 기반)
        gesture = detect_gesture(pose, img_w, img_h)
        self.get_logger().info(f"Gesture={gesture} dist={dist:.2f}m")

        # 5) 제스처 -> Twist 매핑 후 /cmd_vel로 송신
        twist = Twist()
        if gesture == GestureState.FORWARD:
            twist.linear.x = V_FORWARD
        elif gesture == GestureState.BACKWARD:
            twist.linear.x = V_BACKWARD
        elif gesture == GestureState.TURN_LEFT:
            twist.angular.z = +W_TURN
        elif gesture == GestureState.TURN_RIGHT:
            twist.angular.z = -W_TURN
        # STOP/UNKNOWN은 0,0 그대로

        self.cmd_pub.publish(twist)
        self._maybe_stream(img_cuda, f"{gesture} {dist:.2f}m")

    def _publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def _maybe_stream(self, img_cuda, status: str = ""):
        """videoOutput 설정 시 CUDA 프레임을 스트리밍"""
        if not self.video_out:
            return
        self.video_out.Render(img_cuda)
        if status:
            self.video_out.SetStatus(status)
        if not self.video_out.IsStreaming():
            self.get_logger().info("Output stream closed. Shutdown.")
            rclpy.shutdown()

    @staticmethod
    def _pose_area(pose) -> float:
        """포즈의 바운딩 박스 면적(키포인트 min/max 기준)"""
        kps = getattr(pose, "Keypoints", []) or []
        if not kps:
            return 0.0
        xs = [kp.x for kp in kps if hasattr(kp, "x")]
        ys = [kp.y for kp in kps if hasattr(kp, "y")]
        if not xs or not ys:
            return 0.0
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _try_enable_motor_power(self):
        """터틀봇 모터 파워 서비스가 있으면 켜본다."""
        from std_srvs.srv import SetBool

        cli = self.create_client(SetBool, "/motor_power")
        if not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("motor_power service not available")
            return

        req = SetBool.Request()
        req.data = True
        future = cli.call_async(req)

        def _on_response(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info(f"motor_power ON: {resp.message}")
                else:
                    self.get_logger().warn(f"motor_power failed: {resp.message}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"motor_power call failed: {exc}")

        future.add_done_callback(_on_response)

    @staticmethod
    def _get_pose_center(pose) -> Optional[Tuple[float, float]]:
        """poseNet 포즈 keypoints의 평균 좌표를 반환"""
        keypoints = getattr(pose, "Keypoints", None)
        if not keypoints:
            return None
        xs, ys = [], []
        for kp in keypoints:
            try:
                xs.append(kp.x)
                ys.append(kp.y)
            except AttributeError:
                continue
        if not xs:
            return None
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def main():
    # --output 인자로 스트리밍 URI 지정 가능 (예: rtsp://@:8554/live)
    parser = argparse.ArgumentParser(description="parkingbot gesture -> cmd_vel")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="videoOutput URI (예: rtp://<ip>:1234, webrtc://@:8554, rtsp://@:8554/live)",
    )
    parser.add_argument(
        "--enable-motor-power",
        action="store_true",
        help="Start-up에서 /motor_power 서비스를 호출해 모터 전원을 켭니다.",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = ParkingBotNode(output_uri=args.output, enable_motor_power=args.enable_motor_power)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.depth_cam.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
