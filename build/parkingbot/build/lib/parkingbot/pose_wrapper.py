"""
poseNet 래퍼
- 모델 로드와 추론만 담당 (ROS 의존성 없음)
"""
import os
from jetson_inference import poseNet


class PoseEstimator:
    def __init__(self, network_name: str = "resnet18-body"):
        # JETSON_INFERENCE_DATA_DIR가 설정돼 있고 모델 파일이 존재하면 절대 경로로 로드
        data_dir = os.getenv("JETSON_INFERENCE_DATA_DIR")
        pose_dir = os.path.join(data_dir, "networks", "Pose-ResNet18-Body") if data_dir else None
        model_path = os.path.join(pose_dir, "pose_resnet18_body.onnx") if pose_dir else None
        topo_path = os.path.join(pose_dir, "human_pose.json") if pose_dir else None
        colors_path = os.path.join(pose_dir, "colors.txt") if pose_dir else None

        if model_path and os.path.exists(model_path) and os.path.exists(topo_path):
            # 데이터 디렉터리에서 직접 로드 (모델 검색 실패 방지)
            args = [
                f"--model={model_path}",
                f"--topology={topo_path}",
            ]
            if colors_path and os.path.exists(colors_path):
                args.append(f"--colors={colors_path}")
            try:
                self.net = poseNet(argv=args)
                return
            except Exception:
                # 파라미터가 지원되지 않는 버전이면 fallback
                pass
        # 기본 동작: 이름만 넘겨서 poseNet 로드
        self.net = poseNet(network_name)

    def estimate(self, cuda_image, overlay: str = "links,keypoints"):
        """
        cuda_image: jetson_utils videoSource가 반환하는 CUDA 이미지
        overlay: keypoints/links overlay 옵션
        """
        return self.net.Process(cuda_image, overlay=overlay)
