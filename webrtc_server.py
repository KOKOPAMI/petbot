import cv2
import av
import json
import asyncio
import threading
import time
import random
import pyaudio
import queue

# =================================================================
# 🌡️ DTPM151 체온 센서 SPI 설정 구역 - 기존 테스트 코드 기준
# =================================================================
SENSOR_ENABLED = True
is_running = True  # 스레드들의 안전한 종료를 제어하는 전역 플래그

try:
    import os
    os.environ.setdefault("JETSON_MODEL_NAME", "JETSON_ORIN_NANO")

    import spidev
    import Jetson.GPIO as GPIO

    BUS, CS_DEV = 0, 0
    CS_BCM = 8

    SPI_MODE = 3
    SPI_SPEED = 1_000_000

    CMD_OBJ = 0xA0
    CMD_SEN = 0xA1

    def usleep(us):
        time.sleep(us / 1_000_000.0)

    def s16(v):
        return v - 0x10000 if (v & 0x8000) else v

    spi = spidev.SpiDev()
    spi.open(BUS, CS_DEV)
    spi.mode = SPI_MODE
    spi.max_speed_hz = SPI_SPEED

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(CS_BCM, GPIO.OUT, initial=GPIO.HIGH)

    def read16(cmd):
        if not SENSOR_ENABLED:
            return 0

        GPIO.output(CS_BCM, GPIO.LOW)
        usleep(10)

        spi.xfer2([cmd])
        usleep(10)

        lo = spi.xfer2([0x22])[0]
        usleep(10)

        hi = spi.xfer2([0x22])[0]
        usleep(10)

        GPIO.output(CS_BCM, GPIO.HIGH)

        return (hi << 8) | lo

    def read_object_c():
        return round(s16(read16(CMD_OBJ)) / 10.0, 1)

    def read_sensor_c():
        return round(s16(read16(CMD_SEN)) / 10.0, 1)

except Exception as e:
    print(f"⚠️ [하드웨어 알림] 젯슨 SPI 센서를 초기화할 수 없습니다 ({e}). 시뮬레이션 모드로 작동합니다.")
    SENSOR_ENABLED = False

# =================================================================
# YOLO 모델 및 카메라 초기화 (정상 확인된 1번 카메라 고정)
# =================================================================
from ultralytics import YOLO
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack
)
from motor import set_motor_speed

model = YOLO("best (1).pt")
pcs = set()
last_command = "STOP"

cap = cv2.VideoCapture(0)  # 💡 정상 작동 확인된 1번 인덱스 적용
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 20)

detected_object = "NONE"
battery = 82
wifi = "CONNECTED"
robot_state = "ONLINE"

motor_left = 0
motor_right = 0

raw_camera_frame = None      
latest_annotated_frame = None

prev_error = 0

current_state = "MANUAL"  
state_timer = 0          
lost_start_time = 0      
last_turn_dir = 1

pet_temp = 36.5  # 🌡️ 전역 변수

# ==========================================
# 백그라운드 멀티스레드 루프 구역
# ==========================================

def camera_reader_thread():
    global raw_camera_frame, is_running
    print("📸 [1단계] 카메라 버퍼 드레인 스레드 가동 완료")
    while is_running:
        ret, frame = cap.read()
        if ret:
            raw_camera_frame = frame
        else:
            time.sleep(0.01)

def temp_sensor_reader_thread():
    global pet_temp, is_running
   
    if not SENSOR_ENABLED:
        print("🌡️ [3단계] 센서 하드웨어 부재 -> 가상 체온 시뮬레이터 가동 시작")
        while is_running:
            pet_temp = round(36.5 + random.uniform(-0.2, 0.2), 1)
            time.sleep(0.5)
        return

    print("🌡️ [3단계] 젯슨 실물 DTPM151 하드웨어 센서 루프 가동 완료")
    
    # 초기 안정화 센서 예열
    for _ in range(3):
        try:
            read_object_c()
            read_sensor_c()
        except:
            pass
        time.sleep(0.05)
       
    print_counter = 0
    while is_running:
        try:
            if not SENSOR_ENABLED:
                break
                
            tobj = read_object_c()
            tsen = read_sensor_c()
           
            # 💡 [여기 수정] 필터링 조건과 상관없이 무조건 0.5초마다 터미널에 원본을 찍습니다!
            print(f"📡 [RAW 디버깅] 체온(Obj): {tobj:.1f} °C | 센서주변(Sen): {tsen:.1f} °C")
            
            if 15.0 < tobj < 45.0:  
                pet_temp = tobj
               
        except Exception as e:
            # 에러가 나면 숨기지 말고 터미널에 범인을 출력합니다.
            print(f"⚠️ 센서 읽기 실패 원인: {e}")
            
        time.sleep(0.5)

# ==========================================
# YOLO 분석 및 영상 처리 알고리즘 루프
# ==========================================
def process_yolo_and_control(frame, model_engine):
    global current_state, motor_left, motor_right, detected_object
    global state_timer, lost_start_time, last_turn_dir
    global prev_error

    small_frame = cv2.resize(frame, (416, 234))
    results = model_engine(small_frame, imgsz=416, conf=0.65, verbose=False)
    boxes = results[0].boxes
   
    if boxes is not None and len(boxes) > 0:
        cls = int(boxes.cls[0].item())
        target_name = results[0].names[cls]
       
        if target_name in ["Dog", "Cat"]:
            detected_object = target_name
        else:
            detected_object = "NONE"
    else:
        detected_object = "NONE"

    annotated_frame = frame.copy()
    current_time = time.time()

    if current_state == "MANUAL":
        pass

    elif current_state == "EXPLORATION":
        if detected_object != "NONE":
            current_state = "TRACKING"
            print("🎯 [STATE] EXPLORATION -> TRACKING (대상 발견)")
        else:
            action_elapsed = current_time - state_timer
            if action_elapsed < 5.0:
                motor_left, motor_right = 120, 120
            elif action_elapsed < 6.5:
                motor_left = -100 * last_turn_dir
                motor_right = 100 * last_turn_dir
            else:
                state_timer = current_time
                last_turn_dir *= -1
           
            set_motor_speed(motor_left, motor_right)
            cv2.putText(annotated_frame, "MODE: EXPLORATION (PATROL)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif current_state == "TRACKING":
        if detected_object == "NONE":
            current_state = "RECOVERY"
            lost_start_time = current_time
            prev_error = 0
            print("⚠️ [STATE] TRACKING -> RECOVERY (대상 유실)")
        else:
            best_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
           
            cv2.rectangle(annotated_frame, (x1*2, y1*2), (x2*2, y2*2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"TRACKING: {detected_object}", (x1*2, y1*2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_h = small_frame.shape[0]  
            box_height = int(y2 - y1)
            height_ratio = box_height / frame_h  

            center_x = (x1 + x2) // 2
            frame_center = small_frame.shape[1] // 2  
            error = center_x - frame_center

            if abs(error) < 20:
                error = 0

            Kp = 0.38  
            Kd = 0.18  

            derivative = error - prev_error
            control = (Kp * error) + (Kd * derivative)
            control = max(-60, min(60, control))
            prev_error = error

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

            motor_left, motor_right = left, right
            set_motor_speed(motor_left, motor_right)

            cv2.putText(annotated_frame, f"MODE: TRACKING ({distance_status})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    elif current_state == "RECOVERY":
        if detected_object != "NONE":
            current_state = "TRACKING"
            print("🔄 [STATE] RECOVERY -> TRACKING (복구 성공)")
        else:
            lost_elapsed = current_time - lost_start_time
            if lost_elapsed <= 5.0:
                motor_left, motor_right = -90, 90
                set_motor_speed(motor_left, motor_right)
                cv2.putText(annotated_frame, f"MODE: RECOVERY ({5.0 - lost_elapsed:.1f}s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                current_state = "EXPLORATION"
                state_timer = current_time  
                print("■ [STATE] RECOVERY -> EXPLORATION (복구 실패, 순찰 복귀)")

    return annotated_frame

async def camera_inference_loop():
    global raw_camera_frame, latest_annotated_frame, is_running
    print("✅ [2단계] 백그라운드 YOLO 분석 및 모터 제어 루프 가동 시작")
   
    while is_running:
        try:
            if raw_camera_frame is None:
                await asyncio.sleep(0.01)
                continue

            current_frame = raw_camera_frame.copy()
            annotated_frame = await asyncio.to_thread(
                process_yolo_and_control, current_frame, model
            )
            latest_annotated_frame = annotated_frame
            await asyncio.sleep(0.04)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ YOLO 루프 내부 예외 발생: {e}")
            await asyncio.sleep(1)

class CameraTrack(VideoStreamTrack):
    async def recv(self):
        global latest_annotated_frame
        pts, time_base = await self.next_timestamp()

        while latest_annotated_frame is None:
            await asyncio.sleep(0.01)

        frame_to_send = latest_annotated_frame.copy()
        frame_rgb = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

# ==========================================
# 웹 서버 라우팅 (Aiohttp HTTP Endpoints)
# ==========================================
async def index(request):
    with open("index.html", "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/html")

async def monitor(request):
    with open("monitor.html", "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/html")

async def control_page(request):
    with open("control.html", "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/html")

async def style(request):
    with open("style.css", "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/css")

async def control(request):
    global current_state, last_command, state_timer
    global motor_left, motor_right

    data = await request.json()
    command = data.get("command", "STOP")

    if command == last_command:
        return web.Response(text="SKIP")
    last_command = command

    if command == "AUTO_TRACK_ON":
        if detected_object != "NONE":
            current_state = "TRACKING"
        else:
            current_state = "EXPLORATION"
            state_timer = time.time()
        return web.Response(text="AUTO MODE START")

    elif command == "AUTO_TRACK_OFF" or command == "STOP":
        current_state = "MANUAL"
        motor_left, motor_right = 0, 0
        set_motor_speed(0, 0)
        return web.Response(text="MANUAL MODE (STOP)")

    current_state = "MANUAL"
    speed = 160

    if command == "FORWARD":
        motor_left, motor_right = speed, speed
    elif command == "BACK":
        motor_left, motor_right = -speed, -speed
    elif command == "LEFT":
        motor_left, motor_right = -100, 100
    elif command == "RIGHT":
        motor_left, motor_right = 100, -100
    else:
        motor_left, motor_right = 0, 0

    set_motor_speed(motor_left, motor_right)
    return web.Response(text="OK")

audio_queue = queue.Queue(maxsize=2)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
resampler = av.AudioResampler(format='s16', layout='mono', rate=48000)

def audio_playback_worker():
    global is_running
    print("🔊 [오디오 엔진] 전용 스레드가 완벽하게 기동되었습니다.")
    while is_running:
        try:
            audio_data = audio_queue.get(timeout=0.5)
            if audio_data is None:
                continue
            stream.write(audio_data)
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️ 오디오 일꾼 스레드 내부 출력 오류: {e}")
            time.sleep(0.1)

t_audio_worker = threading.Thread(target=audio_playback_worker, daemon=True)
t_audio_worker.start()

# =================================================================
# WebRTC 시그널링 엔드포인트 수정 구역
# =================================================================
async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            print("🗑️ [메모리 수거] 연결 해제된 클라이언트 웹 자원을 완전히 반납했습니다.")

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            print("▶ [WebRTC] 아이폰 마이크 트랙 연결 성공!")
            while is_running:
                try:
                    frame = await track.recv()
                    resampled_frames = resampler.resample(frame)
                   
                    for r_frame in resampled_frames:
                        audio_data = r_frame.to_ndarray().tobytes()
                       
                        while audio_queue.qsize() > 0:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                       
                        audio_queue.put_nowait(audio_data)
                   
                except Exception as e:
                    print("마이크 연결 종료 또는 에러:", e)
                    break

    pc.addTrack(CameraTrack())
   
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def get_status(request):
    global detected_object, current_state, battery, wifi, robot_state, motor_left, motor_right
    global pet_temp  
   
    return web.json_response({
        "battery": battery,
        "wifi": wifi,
        "state": robot_state,
        "detected": detected_object,
        "tracking": current_state != "MANUAL",
        "mode": current_state,                  
        "left": motor_left,
        "right": motor_right,
        "pet_temp": round(pet_temp, 1)  
    })

# ==========================================
# 서버 가동 및 비상 안전 장치 구역
# ==========================================
async def start_background_tasks(app_context):
    t_cam = threading.Thread(target=camera_reader_thread, daemon=True)
    t_cam.start()
   
    t_temp = threading.Thread(target=temp_sensor_reader_thread, daemon=True)
    t_temp.start()
   
    app_context['camera_loop'] = asyncio.create_task(camera_inference_loop())

async def cleanup_background_tasks(app_context):
    app_context['camera_loop'].cancel()
    await app_context['camera_loop']

async def on_shutdown(app_context):
    global SENSOR_ENABLED, is_running
    print("🚨 로봇 관제 웹 서버 정지 절차에 진입합니다.")
    
    is_running = False  # 모든 스레드 루프 중단 유도
    time.sleep(0.2)
    
    try:
        set_motor_speed(0, 0)
    except:
        pass
        
    cap.release()
   
    if SENSOR_ENABLED:
        try:
            SENSOR_ENABLED = False
            spi.close()
        except:
            pass

    close_connections = [pc.close() for pc in pcs]
    if close_connections:
        await asyncio.gather(*close_connections)
    pcs.clear()
    print("👋 자원 및 SPI 통신 자원 반납 정상 완료")

app = web.Application()
app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)

app.router.add_get("/", index)
app.router.add_get("/monitor", monitor)
app.router.add_get("/control_page", control_page)
app.router.add_get("/style.css", style)
app.router.add_post("/offer", offer)
app.router.add_post("/control", control)
app.router.add_get("/status", get_status)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    import ssl
   
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')

    try:
        web.run_app(app, host="0.0.0.0", port=5000, ssl_context=ssl_context)
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 서버 제어가 중단되었습니다 (Ctrl+C).")
    finally:
        print("🚨 [비상 제어] 프로세스 종료를 감지하여 모터를 무조건 안전 정지합니다.")
        is_running = False
        try:
            set_motor_speed(0, 0)
            cap.release()
        except:
            pass
        print("👋 시스템 완전히 안전 종료됨.")