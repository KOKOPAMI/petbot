# tracking.py
import cv2
import time
import numpy as np
from motor import set_motor_speed

class RobotTracker:
    def __init__(self):
        # 자율주행 상태 변수들 격리
        self.current_state = "MANUAL"  
        self.state_timer = 0          
        self.lost_start_time = 0      
        self.last_turn_dir = 1
        self.prev_error = 0
        self.detected_object = "NONE"
        
        # 모터 속도 상태
        self.motor_left = 0
        self.motor_right = 0

    def process_yolo_and_control(self, frame, model_engine):
        current_time = time.time()
        
        # 1. 이미지 리사이즈 및 YOLO 추론
        small_frame = cv2.resize(frame, (416, 234))
        results = model_engine.predict(small_frame, imgsz=416, conf=0.65, verbose=False)
        boxes = results[0].boxes
       
        # 2. 타겟 인식 검사 (Dog, Cat)
        if boxes is not None and len(boxes) > 0:
            cls = int(boxes.cls[0].item())
            target_name = results[0].names[cls]
            if target_name in ["Dog", "Cat"]:
                self.detected_object = target_name
            else:
                self.detected_object = "NONE"
        else:
            self.detected_object = "NONE"

        annotated_frame = frame  # 스트림은 원본 사용, 주석은 상태 API로 확인

        # 3. 상태 머신 (State Machine) 제어
        if self.current_state == "MANUAL":
            pass

        elif self.current_state == "EXPLORATION":
            if self.detected_object != "NONE":
                self.current_state = "TRACKING"
                print("🎯 [STATE] EXPLORATION -> TRACKING (대상 발견)")
            else:
                action_elapsed = current_time - self.state_timer
                if action_elapsed < 5.0:
                    self.motor_left, self.motor_right = 120, 120
                elif action_elapsed < 6.5:
                    self.motor_left = -100 * self.last_turn_dir
                    self.motor_right = 100 * self.last_turn_dir
                else:
                    self.state_timer = current_time
                    self.last_turn_dir *= -1
               
                set_motor_speed(self.motor_left, self.motor_right)
                cv2.putText(annotated_frame, "MODE: EXPLORATION (PATROL)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        elif self.current_state == "TRACKING":
            if self.detected_object == "NONE":
                self.current_state = "RECOVERY"
                self.lost_start_time = current_time
                self.prev_error = 0
                print("⚠️ [STATE] TRACKING -> RECOVERY (대상 유실)")
            else:
                best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
               
                cv2.rectangle(annotated_frame, (x1*2, y1*2), (x2*2, y2*2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"TRACKING: {self.detected_object}", (x1*2, y1*2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame_h = small_frame.shape[0]  
                box_height = int(y2 - y1)
                height_ratio = box_height / frame_h  

                center_x = (x1 + x2) // 2
                frame_center = small_frame.shape[1] // 2  
                error = center_x - frame_center

                if abs(error) < 20:
                    error = 0

                # PID 제어 루프
                Kp = 0.38  
                Kd = 0.18  

                derivative = error - self.prev_error
                control = (Kp * error) + (Kd * derivative)
                control = max(-60, min(60, control))
                self.prev_error = error

                if height_ratio < 0.25:
                    forward = 120
                    distance_status = "FAR (APPROACH)"
                elif height_ratio < 0.45:
                    forward = 60
                    distance_status = "MID (SLOW DOWN)"
                elif height_ratio < 0.60:
                    forward = 0
                    distance_status = "ARRIVED (STOP)"
                    if abs(error) < 40:
                        control = 0
                else:
                    forward = -60
                    distance_status = "TOO CLOSE (BACKUP)"

                left = int(max(-255, min(255, forward + control)))
                right = int(max(-255, min(255, forward - control)))

                MIN_SPEED = 90
               
                if left != 0 and abs(left) < MIN_SPEED:
                    left = MIN_SPEED if left > 0 else -MIN_SPEED
                if right != 0 and abs(right) < MIN_SPEED:
                    right = MIN_SPEED if right > 0 else -MIN_SPEED

                self.motor_left, self.motor_right = left, right
                set_motor_speed(self.motor_left, self.motor_right)

                cv2.putText(annotated_frame, f"MODE: TRACKING ({distance_status})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elif self.current_state == "RECOVERY":
            if self.detected_object != "NONE":
                self.current_state = "TRACKING"
                print("🔄 [STATE] RECOVERY -> TRACKING (복구 성공)")
            else:
                lost_elapsed = current_time - self.lost_start_time
                if lost_elapsed <= 5.0:
                    self.motor_left, self.motor_right = -90, 90
                    set_motor_speed(self.motor_left, self.motor_right)
                    cv2.putText(annotated_frame, f"MODE: RECOVERY ({5.0 - lost_elapsed:.1f}s)", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    self.current_state = "EXPLORATION"
                    self.state_timer = current_time  
                    print("■ [STATE] RECOVERY -> EXPLORATION (복구 실패, 순찰 복귀)")

        return annotated_frame