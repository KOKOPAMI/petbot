import serial
import time
import threading
import queue

ser = None
_motor_queue = queue.Queue(maxsize=1)
_last_sent_left = None
_last_sent_right = None
_last_error_log = 0.0
_last_send_time = 0.0
MOTOR_SEND_INTERVAL = 0.08  # 초당 최대 ~12회

try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1, write_timeout=0.3)
    time.sleep(2)
    print("✅ 아두이노 모터 컨트롤러 고속 연결 성공")
except serial.SerialException:
    print("⚠️ 경고: 아두이노가 연결되지 않았습니다. (테스트 모드로 동작합니다)")
except Exception as e:
    print(f"⚠️ 시리얼 연결 중 알 수 없는 오류 발생: {e}")


def _motor_worker():
    global _last_sent_left, _last_sent_right, _last_error_log, _last_send_time

    while True:
        try:
            left, right = _motor_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        while True:
            try:
                left, right = _motor_queue.get_nowait()
            except queue.Empty:
                break

        if ser is None or not ser.is_open:
            if left != _last_sent_left or right != _last_sent_right:
                _last_sent_left = left
                _last_sent_right = right
                _last_send_time = time.time()
                print(f"MOTOR COMMAND -> LEFT: {left}, RIGHT: {right} (시뮬레이션)")
            continue

        wait = MOTOR_SEND_INTERVAL - (time.time() - _last_send_time)
        if wait > 0:
            time.sleep(wait)

        if left == _last_sent_left and right == _last_sent_right:
            continue

        try:
            ser.write(f"{left},{right}\n".encode())
            _last_sent_left = left
            _last_sent_right = right
            _last_send_time = time.time()
            print(f"MOTOR COMMAND -> LEFT: {left}, RIGHT: {right}")
        except Exception as e:
            now = time.time()
            if now - _last_error_log >= 5.0:
                print(f"⚠️ 모터 데이터 전송 실패: {e}")
                _last_error_log = now


_motor_thread = threading.Thread(target=_motor_worker, daemon=True)
_motor_thread.start()


def set_motor_speed(left_speed, right_speed):
    left_speed = int(left_speed)
    right_speed = int(right_speed)

    if _motor_queue.full():
        try:
            _motor_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        _motor_queue.put_nowait((left_speed, right_speed))
    except queue.Full:
        pass
