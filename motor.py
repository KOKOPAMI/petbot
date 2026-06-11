import serial
import time

ser = None

try:
    # 💡 아두이노 스케치와 동기화하기 위해 115200 고속 보레이트 적용
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2) 
    print("✅ 아두이노 모터 컨트롤러 고속 연결 성공")
except serial.SerialException:
    print("⚠️ 경고: 아두이노가 연결되지 않았습니다. (테스트 모드로 동작합니다)")
except Exception as e:
    print(f"⚠️ 시리얼 연결 중 알 수 없는 오류 발생: {e}")

def set_motor_speed(left_speed, right_speed):
    data = f"{left_speed},{right_speed}\n"
    
    if ser is not None and ser.is_open:
        try:
            ser.write(data.encode())
        except Exception as e:
            print(f"⚠️ 모터 데이터 전송 실패: {e}")
            
    print(f"MOTOR COMMAND -> LEFT: {left_speed}, RIGHT: {right_speed}")