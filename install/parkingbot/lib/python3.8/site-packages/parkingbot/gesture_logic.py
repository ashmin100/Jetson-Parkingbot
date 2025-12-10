"""
개선된 제스처 인식기 (poseNet 기반)
- keypoint confidence 검사
- 어깨 거리 기반 정규화(scale-invariant)
- 어깨 기울기 보정(좌표계 회전)
- 독립적인 제스처 판별 함수
- 프레임 안정화(smoothing)
"""

from typing import Optional, Tuple, List, Dict
import math


# -------------------------------
# 상수 정의
# -------------------------------
# PoseNet keypoint IDs for head region
HEAD_KEYPOINT_IDS = {
    0,   # nose
    1,   # left_eye
    2,   # right_eye
    3,   # left_ear
    4,   # right_ear
    17   # neck
}


# -------------------------------
# 제스처 상태
# -------------------------------
class GestureState:
    STOP = "STOP"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    UNKNOWN = "UNKNOWN"


# -------------------------------
# 파라미터 (정규화된 거리 기반)
# -------------------------------
class GestureParams:
    horiz_tol = 0.50       # 양팔 수평 tolerance (정규화)
    side_th = 0.40         # 옆으로 뻗은 거리 threshold
    down_th = 0.20         # 팔 내린 기준
    turn_down_th = 0.15    # 좌/우 회전 시 반대팔 내려간 기준 (완화)
    up_th = 0.60           # 팔 올린 기준
    head_margin = 0.05     # 머리 높이 대비 여유
    min_conf = 0.25        # keypoint confidence 최소값


params = GestureParams()


# -------------------------------
# Keypoint extraction
# -------------------------------
def _find_keypoints(pose):
    """어깨/손목 + confidence 가져오기"""
    kps = getattr(pose, "Keypoints", []) or []

    def _kp_name(k) -> str:
        if hasattr(k, "Name"):
            return (k.Name or "").lower()
        if hasattr(k, "name"):
            return (k.name or "").lower()
        return ""

    by_name: Dict[str, object] = {}
    by_id: Dict[int, object] = {}
    for k in kps:
        nm = _kp_name(k)
        if nm:
            by_name[nm] = k
        if hasattr(k, "ID"):
            by_id[k.ID] = k

    def pick(names, ids):
        for n in names:
            if n in by_name:
                return by_name[n]
        for i in ids:
            if i in by_id:
                return by_id[i]
        return None

    L_sh = pick(["left_shoulder", "l_shoulder", "left shoulder"], [5])
    R_sh = pick(["right_shoulder", "r_shoulder", "right shoulder"], [6])
    L_wr = pick(["left_wrist", "l_wrist", "left wrist"], [9])
    R_wr = pick(["right_wrist", "r_wrist", "right wrist"], [10])

    if not (L_sh and R_sh and L_wr and R_wr):
        return None

    # confidence check
    if (getattr(L_sh, "score", 1.0) < params.min_conf or
        getattr(R_sh, "score", 1.0) < params.min_conf or
        getattr(L_wr, "score", 1.0) < params.min_conf or
        getattr(R_wr, "score", 1.0) < params.min_conf):
        return None

    return L_sh, R_sh, L_wr, R_wr


# -------------------------------
# 좌표 회전 및 정규화
# -------------------------------
def rotate_point(x, y, cx, cy, angle):
    """(cx,cy) 기준 angle rad 만큼 회전"""
    dx, dy = x - cx, y - cy
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    return rx, ry


def normalize_keypoints(L_sh, R_sh, L_wr, R_wr):
    """
    - 어깨를 기준으로 좌표계 회전 (수평 정렬)
    - 어깨 거리로 정규화 (scale-invariant)
    반환: (L_sh_norm, R_sh_norm, L_wr_norm, R_wr_norm) - 각각 (x, y) 튜플
    """
    shoulder_dx = R_sh.x - L_sh.x
    shoulder_dy = R_sh.y - L_sh.y
    angle = -math.atan2(shoulder_dy, shoulder_dx)

    cx = (L_sh.x + R_sh.x) / 2
    cy = (L_sh.y + R_sh.y) / 2
    scale = math.hypot(shoulder_dx, shoulder_dy)

    def norm(kp):
        rx, ry = rotate_point(kp.x, kp.y, cx, cy, angle)
        return rx / scale, ry / scale

    return norm(L_sh), norm(R_sh), norm(L_wr), norm(R_wr)


# -------------------------------
# 독립적인 제스처 판별 함수들
# -------------------------------
def check_backward(L_wr_y: float, R_wr_y: float, head_y: Optional[float]) -> float:
    """
    BACKWARD 제스처 확신도 계산 (양팔을 머리 위로)
    반환: 0.0 이상의 점수 (높을수록 확신도 높음), 조건 불만족 시 0.0
    """
    if head_y is not None:
        # 머리 위치 기준으로 양팔이 얼마나 위에 있는지 (y축 반전)
        left_above = L_wr_y - head_y + params.head_margin
        right_above = R_wr_y - head_y + params.head_margin
        
        # 양팔 모두 머리 위에 있어야 함
        if left_above > 0 and right_above > 0:
            return left_above + right_above
    else:
        # head_y가 없는 경우 절대 위치 기준 (y축 반전)
        if L_wr_y > params.up_th and R_wr_y > params.up_th:
            return (L_wr_y - params.up_th) + (R_wr_y - params.up_th)
    
    return 0.0


def check_forward(L_sh_y: float, R_sh_y: float, 
                 L_wr_x: float, L_wr_y: float, 
                 R_wr_x: float, R_wr_y: float) -> float:
    """
    FORWARD 제스처 확신도 계산 (양팔을 수평으로 옆으로 벌림)
    반환: 0.0 이상의 점수 (높을수록 확신도 높음), 조건 불만족 시 0.0
    """
    # 왼팔: 수평 정렬 정도
    left_is_horizontal = abs(L_wr_y - L_sh_y) < params.horiz_tol
    # 왼팔: 왼쪽으로 벌린 정도
    left_extended_left = L_wr_x < -params.side_th
    
    # 오른팔: 수평 정렬 정도
    right_is_horizontal = abs(R_wr_y - R_sh_y) < params.horiz_tol
    # 오른팔: 오른쪽으로 벌린 정도
    right_extended_right = R_wr_x > params.side_th
    
    # 양팔 모두 조건 만족해야 함
    if left_is_horizontal and left_extended_left and right_is_horizontal and right_extended_right:
        # 수평 정렬 정도 + 벌린 거리 합산
        horiz_score = (params.horiz_tol - abs(L_wr_y - L_sh_y)) + \
                     (params.horiz_tol - abs(R_wr_y - R_sh_y))
        extend_score = (-L_wr_x - params.side_th) + (R_wr_x - params.side_th)
        return horiz_score + extend_score
    
    return 0.0


def check_turn_right(R_sh_y: float, R_wr_x: float, R_wr_y: float, 
                    L_wr_y: float) -> float:
    """
    TURN_RIGHT 제스처 확신도 계산 (오른팔만 수평으로 벌림, 왼팔은 내림)
    반환: 0.0 이상의 점수 (높을수록 확신도 높음), 조건 불만족 시 0.0
    """
    # 오른팔: 수평 정렬
    right_is_horizontal = abs(R_wr_y - R_sh_y) < params.horiz_tol
    # 오른팔: 오른쪽으로 벌림
    right_extended_right = R_wr_x > params.side_th
    # 왼팔: 내려져 있음 (y축 반전 적용)
    left_is_down = L_wr_y < -params.turn_down_th
    
    if right_is_horizontal and right_extended_right and left_is_down:
        # 수평 정렬 + 벌린 거리 + 왼팔 내린 정도
        horiz_score = params.horiz_tol - abs(R_wr_y - R_sh_y)
        extend_score = R_wr_x - params.side_th
        down_score = -L_wr_y - params.turn_down_th
        return horiz_score + extend_score + down_score
    
    return 0.0


def check_turn_left(L_sh_y: float, L_wr_x: float, L_wr_y: float, 
                   R_wr_y: float) -> float:
    """
    TURN_LEFT 제스처 확신도 계산 (왼팔만 수평으로 벌림, 오른팔은 내림)
    반환: 0.0 이상의 점수 (높을수록 확신도 높음), 조건 불만족 시 0.0
    """
    # 왼팔: 수평 정렬
    left_is_horizontal = abs(L_wr_y - L_sh_y) < params.horiz_tol
    # 왼팔: 왼쪽으로 벌림
    left_extended_left = L_wr_x < -params.side_th
    # 오른팔: 내려져 있음 (y축 반전 적용)
    right_is_down = R_wr_y < -params.turn_down_th
    
    if left_is_horizontal and left_extended_left and right_is_down:
        # 수평 정렬 + 벌린 거리 + 오른팔 내린 정도
        horiz_score = params.horiz_tol - abs(L_wr_y - L_sh_y)
        extend_score = -L_wr_x - params.side_th
        down_score = -R_wr_y - params.turn_down_th
        return horiz_score + extend_score + down_score
    
    return 0.0


def check_stop(L_wr_y: float, R_wr_y: float) -> float:
    """
    STOP 제스처 확신도 계산 (양팔을 모두 내림)
    반환: 0.0 이상의 점수 (높을수록 확신도 높음), 조건 불만족 시 0.0
    """
    # 양팔 모두 내려져 있어야 함 (y축 반전)
    if L_wr_y < -params.down_th and R_wr_y < -params.down_th:
        return (-L_wr_y - params.down_th) + (-R_wr_y - params.down_th)
    
    return 0.0


# -------------------------------
# 단일 프레임 제스처 판별
# -------------------------------
def detect_gesture_single_frame(pose) -> str:
    kps = _find_keypoints(pose)
    if not kps:
        return GestureState.UNKNOWN

    L_sh, R_sh, L_wr, R_wr = kps
    
    # 좌표 정규화
    (L_sh_x, L_sh_y), (R_sh_x, R_sh_y), \
    (L_wr_x, L_wr_y), (R_wr_x, R_wr_y) = normalize_keypoints(L_sh, R_sh, L_wr, R_wr)

    # 머리 위치 추출
    head_ys = []
    shoulder_dx = R_sh.x - L_sh.x
    shoulder_dy = R_sh.y - L_sh.y
    angle = -math.atan2(shoulder_dy, shoulder_dx)
    cx = (L_sh.x + R_sh.x) / 2
    cy = (L_sh.y + R_sh.y) / 2
    scale = math.hypot(shoulder_dx, shoulder_dy)
    
    for kp in getattr(pose, "Keypoints", []) or []:
        if getattr(kp, "ID", -1) in HEAD_KEYPOINT_IDS:
            rx, ry = rotate_point(kp.x, kp.y, cx, cy, angle)
            head_ys.append(ry / scale)
    
    head_y = min(head_ys) if head_ys else None

    # 각 제스처를 독립적으로 판별
    scores = {
        GestureState.BACKWARD: check_backward(L_wr_y, R_wr_y, head_y),
        GestureState.FORWARD: check_forward(L_sh_y, R_sh_y, L_wr_x, L_wr_y, R_wr_x, R_wr_y),
        GestureState.TURN_RIGHT: check_turn_right(R_sh_y, R_wr_x, R_wr_y, L_wr_y),
        GestureState.TURN_LEFT: check_turn_left(L_sh_y, L_wr_x, L_wr_y, R_wr_y),
        GestureState.STOP: check_stop(L_wr_y, R_wr_y),
    }
    
    # 점수가 0보다 큰 제스처들 중 최고 점수 선택
    valid_scores = {k: v for k, v in scores.items() if v > 0}
    
    if valid_scores:
        return max(valid_scores, key=valid_scores.get)
    
    # 어떤 제스처도 인식되지 않으면 UNKNOWN 반환
    return GestureState.UNKNOWN


# -------------------------------
# 프레임 안정화 (히스토리 기반)
# -------------------------------
class GestureFilter:
    def __init__(self, window=5):
        self.window = window
        self.buf: List[str] = []

    def update(self, g: str) -> str:
        self.buf.append(g)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        
        # 빈 버퍼 처리
        if not self.buf:
            return GestureState.UNKNOWN
        
        # "최다 등장 제스처"를 최종값으로 사용
        return max(set(self.buf), key=self.buf.count)


gesture_filter = GestureFilter(window=5)


def detect_gesture(pose, img_width=None, img_height=None) -> str:
    """
    최종 안정화된 제스처 출력.
    img_width/img_height 인자는 이전 버전 호환을 위해 남겨두었으며 사용하지 않는다.
    """
    g = detect_gesture_single_frame(pose)
    return gesture_filter.update(g)
