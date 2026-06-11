from ultralytics import YOLO
import cv2
from motor import set_motor_speed
from utils import get_direction

def run_tracking():

    # YOLO 모델 로드
    model = YOLO("best (1).pt")

    # 제어 파라미터
    Kp = 0.3    # 회전 민감도
    base_speed = 120 # 기본 전진 속도
    min_area = 20000 # 거리 유지 기준
    max_area = 50000

    # YOLO Tracking 시작
    results = model.track(
        source=0,
        stream=True,
        persist=True,
        conf=0.5,
        imgsz=224,
        tracker="bytetrack.yaml"
    )

    # 모터 이전 상태 변수 루프 밖으로 이동
    prev_left = -1
    prev_right = -1

    # 잃어버린 시간 체크용 변수 (재탐색 로직용)
    lost_time = 0

    for result in results:
        frame = result.orig_img
        
        # [수정 1] 프레임 크기 가져오기를 루프 내부로 이동
        frame_height, frame_width = frame.shape[:2]
        frame_center = frame_width // 2

        boxes = result.boxes

        # 객체가 하나도 없을 경우 (재탐색 로직 적용)
        if boxes is None or len(boxes) == 0:
            cv2.putText(frame, "NO DETECTION - SEARCHING", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # [고도화] 단순 정지가 아닌, 제자리 회전으로 주변 탐색
            # 예: 양쪽 바퀴를 반대로 돌려 제자리 회전
            left_speed, right_speed = -80, 80 
            
            if left_speed != prev_left or right_speed != prev_right:
                set_motor_speed(left_speed, right_speed)
                prev_left = left_speed
                prev_right = right_speed

            cv2.imshow("PetBot Tracking", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # --- 이후 로직은 기존과 동일 (가장 큰 객체 선택 및 P-Control) ---
        best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        error = center_x - frame_center
        control = Kp * error

        if area < min_area:
            forward_control = base_speed
        elif area > max_area:
            forward_control = 0
        else:
            forward_control = 80

        left_speed = int(max(-255, min(255, forward_control + control)))
        right_speed = int(max(-255, min(255, forward_control - control)))

        if left_speed != prev_left or right_speed != prev_right:
            set_motor_speed(left_speed, right_speed)
            prev_left = left_speed
            prev_right = right_speed

        print(f"ERROR : {error}")
        print(f"LEFT : {left_speed}")
        print(f"Right : {right_speed}")

        # 시각화

        # 객체 박스
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 객체 중심점
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # 방향 표시
        direction = get_direction(center_x, frame_center, deadzone=50)

        # 방향 텍스트 표시
        cv2.putText(
            frame,
            direction,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 중심 좌표 표시
        cv2.putText(
            frame,
            f"X: {center_x}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # 바퀴 속도 표시
        cv2.putText(
            frame,
            f"L:{left_speed} R:{right_speed}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        
        # =========================
        # 로그 출력
        # =========================
        print("=" * 40)
        print(f"DIRECTION : {direction}")
        print(f"ERROR     : {error}")
        print(f"AREA      : {area}")
        print(f"LEFT      : {left_speed}")
        print(f"RIGHT     : {right_speed}")

        # ESC 키 종료
        if cv2.waitKey(1) == 27:
            break

    # 종료
    cv2.destroyAllWindows()