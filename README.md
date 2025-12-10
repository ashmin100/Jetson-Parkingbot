# parkingbot (Jetson Xavier + RealSense + TurtleBot3)

## 1. 프로젝트 개요
- Jetson Xavier, Intel RealSense, TurtleBot3를 이용해 사람 팔 제스처로 로봇을 제어하는 실습용 프로젝트입니다.
- RGB 카메라로 poseNet(resnet18-body) 스켈레톤을 추정하고, 어깨 회전 보정·정규화·confidence 필터·프레임 스무딩을 거친 휴리스틱으로 멈춤/전진/후진/좌회전/우회전을 수행합니다.
- RealSense 깊이로 사람이 일정 거리(기본 0.0~3.0m) 안에 있을 때만 제스처를 인정해 안전성을 높였습니다.
- 지능형 로봇 수업 과제/랩 실습 수준으로 직관적이고 단순한 코드 구조를 지향합니다.

## 2. 전체 시스템 구조
RealSense RGB → poseNet → 스켈레톤 + 키포인트(가장 큰 포즈 선택)  
RealSense Depth → 거리 계산  
Gesture Logic(정규화+스무딩) → `/cmd_vel` Twist → TurtleBot3 이동  
옵션: jetson_utils `videoOutput`로 overlay 스트리밍

## 3. 파일 구조 (ament_python)
```
parkingbot/
├── package.xml
├── setup.py
├── parkingbot/
│   ├── __init__.py
│   ├── vision_node.py    # ROS2 노드, 센서 캡처, 제스처→/cmd_vel
│   ├── gesture_logic.py  # 제스처 판별 순수 함수와 threshold 상수
│   ├── depth_helper.py   # RealSense depth 파이프라인, 정렬, 거리 조회
│   └── pose_wrapper.py   # poseNet 로드/추론 래퍼
└── launch/
    └── parkingbot.launch.py
```
- `vision_node.py`: `ParkingBotNode`에서 10Hz 타이머로 RGB 캡처 → poseNet 추론 → depth 거리 확인 → `detect_gesture` → `/cmd_vel` 퍼블리시. `--output`으로 videoOutput 스트리밍 가능.
- `gesture_logic.py`: 어깨 회전 보정·정규화된 좌표계에서 독립 점수로 5개 제스처 판별, ROS 의존성 없음.
- `depth_helper.py`: `DepthCamera`로 depth/color 정렬(`rs.align`), `get_distance`로 픽셀 깊이(m) 반환.
- `pose_wrapper.py`: `PoseEstimator`로 poseNet(resnet18-body) 로드 및 `estimate`.
- `launch/parkingbot.launch.py`: `parkingbot_node` 실행용 launch.

## 4. 제스처 처리 로직 (parkingbot/gesture_logic.py)
- 핵심 변수/좌표계
  - `min_conf=0.25`: poseNet 키포인트 신뢰도. 이보다 낮으면 제스처 판단 자체를 하지 않음.
  - `horiz_tol=0.50`: “팔이 어깨 높이에 가깝다”를 허용하는 y 오차(정규화 단위).
  - `side_th=0.40`: “좌우로 충분히 뻗었다”로 인정하는 x 거리(어깨 길이 기준).
  - `down_th=0.20`, `turn_down_th=0.15`: 팔을 내렸다고 보는 y 임계값(회전/정규화 좌표에서 음수 방향).
  - `up_th=0.60`: 머리 정보를 못 찾았을 때 팔을 올린 것으로 간주하는 절대 y 기준.
  - `head_margin=0.05`: 머리보다 얼마나 더 위여야 BACKWARD로 인정할지 여유값.
  - 좌표계: 왼어깨→오른어깨가 +x, 어깨 중점이 원점. 어깨를 수평으로 회전 보정하고 어깨 길이로 나눠 스케일을 통일. y는 위로 +, 아래로 - (어깨 높이 ≈ 0).
- 처리 흐름
  1) 키포인트 모으기: 어깨·손목을 name/ID(5,6,9,10)로 찾고 `min_conf` 미만이면 UNKNOWN 리턴.
  2) 좌표 정규화: 회전 보정+스케일 정규화로 자세가 기울거나 거리가 달라도 동일 기준에서 비교.
  3) 머리 높이 계산: HEAD_KEYPOINT_IDS(0,1,2,3,4,17) y 최솟값을 `head_y`로 저장(팔이 머리 위인지 판단용).
  4) 제스처 점수 계산: 각 제스처마다 조건을 만족하면 양수 점수를 주고, 그중 가장 큰 값을 선택(모두 0이면 UNKNOWN).
  5) 스무딩: `GestureFilter(window=5)`가 최근 5프레임 최빈값을 반환해 순간 노이즈를 제거(`detect_gesture`).
- 제스처 판별 기준(모두 정규화 좌표 기준)
  - BACKWARD = 양팔을 머리 위로: 두 손목이 `head_y + head_margin`보다 위. 머리 정보가 없으면 `up_th=0.60` 이상이면 인정.
  - FORWARD = 양팔을 좌우로 수평 벌림: 두 손목이 각 어깨 높이에 가깝게(`|Δy| < horiz_tol`) 있고, 왼손목은 왼쪽으로 `side_th` 이상, 오른손목은 오른쪽으로 `side_th` 이상. 수평성+벌림 거리를 합산해 점수화.
  - TURN_RIGHT = 오른팔만 수평, 왼팔 내림: 오른손목이 어깨 높이 근처(`|Δy| < horiz_tol`), 오른쪽으로 `side_th` 이상, 왼손목은 `-turn_down_th` 아래.
  - TURN_LEFT = 왼팔만 수평, 오른팔 내림: 왼손목이 어깨 높이 근처, 왼쪽으로 `side_th` 이상, 오른손목은 `-turn_down_th` 아래.
  - STOP = 양팔 내림: 두 손목이 `-down_th` 아래.

## 5. Depth 사용 방식
- depth/color를 640×480@30fps로 정렬(`rs.align`)해 RGB 키포인트와 픽셀을 맞춥니다.
- 포즈 키포인트 평균 좌표를 중심 픽셀로 잡아 depth를 읽고 `depth_scale`을 곱해 미터 단위로 변환합니다.
- 거리 범위(`MIN_DIST=0.0` ~ `MAX_DIST=3.0`) 밖이면 STOP을 발행하고 스트리밍 상태창에 거리 정보를 표시합니다.

## 6. ROS2 동작 방식
- 구독 없음(센서 직접 사용), `/cmd_vel`만 퍼블리시(geometry_msgs/Twist).
- 10Hz 타이머: RGB 캡처 → (여러 포즈 중 바운딩 박스가 가장 큰 포즈 선택) 포즈 추정 → depth 거리 확인 → 제스처 판별/스무딩 → Twist 매핑 → `/cmd_vel`.
- launch(`parkingbot.launch.py`)로 노드 실행, 필요 시 `--output`으로 스트리밍 URI 전달. `--enable-motor-power`로 `/motor_power` 서비스가 있을 때 모터를 켭니다.

## 7. 실행 방법
1) ROS2 Foxy 설치(Python 3.8, Ubuntu 20.04/JetPack 5.x 또는 4.x).  
2) `pyrealsense2` 및 jetson-inference/jetson-utils 환경 준비.  
3) 빌드/실행:
```bash
cd ~/parkingbot
colcon build
source install/setup.bash
ros2 launch parkingbot parkingbot.launch.py
```
- overlay 스트리밍/모터 파워 예:
```bash
ros2 run parkingbot parkingbot_node --output rtsp://@:8554/live --enable-motor-power
```

### Jetson + conda(예: jetson07) 환경 변수 예시
- ROS: `source /opt/ros/foxy/setup.bash`
- conda 활성화 후 파이썬 경로: `export PYTHONPATH=/home/<user>/miniforge3/envs/jetson07/lib/python3.8/site-packages:$PYTHONPATH`
- libgomp TLS 오류(“cannot allocate memory in static TLS block”)가 날 때: `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`

## 8. 보완해야 할 점
- RGB `videoSource` 해상도를 depth(640×480, 30fps)와 꼭 맞춰 사용해야 함.
- `DepthCamera.stop()` 등 종료 정리 루틴 추가 필요.
- RealSense를 V4L2(`/dev/video4`)와 librealsense로 동시에 열 때 드라이버 충돌 가능성 확인 필요.
- 제스처 파라미터(`horiz_tol`, `side_th`, `down_th`, `turn_down_th`, `up_th`, `head_margin`, `min_conf`)와 `GestureFilter` window는 사용자/카메라 각도에 맞춰 추가 튜닝 필요.
- 확장: RGB-D 기반 poseNet 재학습, 다른 포즈 모델로 교체 가능하도록 구조를 더 일반화할 수 있음.

## 9. 전체 요약
- parkingbot은 RealSense RGB+Depth와 poseNet을 사용해 사람 팔 제스처를 인식하고 TurtleBot3를 `/cmd_vel`로 제어하는 ROS2 노드입니다(ROS2 Foxy/Python 3.8 기준).
- 어깨 회전 보정·정규화된 좌표계와 confidence 필터, 머리 높이 기반 후진 검출, 프레임 히스토리 스무딩으로 5개 제스처(멈춤/좌/우/전/후)를 판단합니다.
- 빌드 후 launch로 실행하면 10Hz로 제스처를 읽어 로봇을 움직이며, 필요 시 스트리밍으로 overlay를 확인할 수 있습니다. 향후 해상도/파라미터 튜닝, 종료 처리, 안정성 개선을 진행하면 더 견고한 시스템이 됩니다.

## 10. 부록: 시행착오 기록
- 어깨 기울어진 자세에서 팔 높이 비교가 깨져 회전 보정과 어깨 길이 정규화를 추가했습니다.
- 카메라 거리 변화에 따라 threshold가 달라지는 문제를 어깨 길이 기반 scale-invariant 좌표로 해결했습니다.
- 머리 키포인트 누락 시 BACKWARD가 안 잡히는 문제를 `head_y` 유무에 따라 조건을 분기(없으면 `up_th` 절대 기준)해 완화했습니다.
- 낮은 confidence 키포인트로 인한 노이즈를 줄이기 위해 `min_conf=0.25` 미만은 UNKNOWN으로 필터링했습니다.
- 프레임 단위 출렁임으로 오동작이 발생해 5프레임 최빈값 스무딩(`GestureFilter`)을 넣었습니다.
- 여러 사람이 검출되면 제스처가 섞이는 문제를 포즈 바운딩 박스 면적이 가장 큰 사람만 선택하도록 수정했습니다.
- depth/color 정렬이 어긋나 거리 판정이 튀는 문제를 `rs.align` 강제와 poseNet 입력 해상도(640×480) 맞춤으로 해결했습니다.
