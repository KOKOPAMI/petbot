import cv2
import av
import json
import asyncio
import pyaudio
import queue
import time
import threading
import subprocess
from aiohttp import web

# 🌟 login.py 파일에서 안전하게 다중 사용자 인증 기능들을 가져옵니다.
from login import init_db, auth_middleware, login_handler, register_handler

# sensor.py 호출 및 센서 인스턴스 생성
from sensor import TemperatureSensor
is_running = True
sensor_driver = TemperatureSensor()

# tracking.py 호출
from tracking import RobotTracker
tracker = RobotTracker()

# =================================================================
# YOLO 모델 및 카메라 초기화
# =================================================================
from ultralytics import YOLO
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack
)
from motor import set_motor_speed

model = YOLO("best (1).pt")
pcs = set()
last_command = "STOP"

cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 20)

# 서버 시작 시 login 모듈의 데이터베이스 초기화 및 테이블 생성 가동
init_db()

# =================================================================
# 스레드 안전한 비동기 큐 (최신 프레임 1개만 유지)
# =================================================================
frame_queue = asyncio.Queue(maxsize=1)

battery = 82
wifi = "CONNECTED"
robot_state = "ONLINE"
latest_annotated_frame = None

shared_status_cache = {
    "battery": 82,
    "wifi": "Wi-Fi",
    "state": "ONLINE",
    "detected": "NONE",
    "tracking": False,
    "mode": "MANUAL",
    "left": 0,
    "right": 0,
    "pet_temp": 36.5
}

# ==========================================
# 백그라운드 스레드 및 루프 고도화
# ==========================================
def camera_reader_thread(loop):
    global is_running
    print("📸 [1단계] 카메라 버퍼 드레인 스레드 가동 완료")
    while is_running:
        ret, frame = cap.read()
        if ret:
            asyncio.run_coroutine_threadsafe(push_frame_to_queue(frame), loop)
        else:
            time.sleep(0.01)

async def push_frame_to_queue(frame):
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await frame_queue.put(frame)

async def camera_inference_loop():
    global latest_annotated_frame, is_running, shared_status_cache
    print("✅ [2단계] 백그라운드 YOLO 분석 및 모터 제어 루프 가동 시작")
    
    while is_running:
        try:
            current_frame = await frame_queue.get()
            
            annotated_frame = await asyncio.to_thread(
                tracker.process_yolo_and_control, 
                current_frame, 
                model
            )
            
            latest_annotated_frame = annotated_frame
            frame_queue.task_done()

            shared_status_cache["detected"] = tracker.detected_object
            shared_status_cache["tracking"] = tracker.current_state != "MANUAL"
            shared_status_cache["mode"] = tracker.current_state
            shared_status_cache["left"] = tracker.motor_left
            shared_status_cache["right"] = tracker.motor_right
            shared_status_cache["pet_temp"] = round(sensor_driver.pet_temp, 1)

            await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ YOLO 루프 내부 예외 발생: {e}")
            await asyncio.sleep(1)

async def memory_leak_cleaner_loop():
    import gc
    global is_running
    while is_running:
        await asyncio.sleep(60)
        gc.collect()

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

async def settings(request):
    with open("settings.html", "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/html")

async def control(request):
    global last_command
    data = await request.json()
    command = data.get("command", "STOP")

    if command == last_command:
        return web.Response(text="SKIP")
    last_command = command

    if command == "AUTO_TRACK_ON":
        if tracker.detected_object != "NONE": 
            tracker.current_state = "TRACKING"
        else:
            tracker.current_state = "EXPLORATION"
            tracker.state_timer = time.time()
        return web.Response(text="AUTO MODE START")

    elif command == "AUTO_TRACK_OFF" or command == "STOP" or command == "EMERGENCY_STOP":
        tracker.current_state = "MANUAL"      
        tracker.motor_left, tracker.motor_right = 0, 0
        set_motor_speed(0, 0)
        return web.Response(text="MANUAL MODE (STOP)")

    tracker.current_state = "MANUAL"      
    speed = 160

    if command == "FORWARD":
        tracker.motor_left, tracker.motor_right = speed, speed
    elif command == "BACK" or command == "BACKWARD":
        tracker.motor_left, tracker.motor_right = -speed, -speed
    elif command == "LEFT":
        tracker.motor_left, tracker.motor_right = -100, 100
    elif command == "RIGHT":
        tracker.motor_left, tracker.motor_right = 100, -100
    else:
        tracker.motor_left, tracker.motor_right = 0, 0

    set_motor_speed(tracker.motor_left, tracker.motor_right)
    return web.Response(text="OK")

# 🚪 로그아웃 처리: 브라우저의 명찰 쿠키를 만료시켜 제거합니다.
async def logout_handler(request):
    response = web.Response(text="OK")
    response.del_cookie("session_user")
    return response

# 💻 원격 재부팅 처리: 파이썬이 리눅스 시스템에 직접 sudo reboot 명령을 전송합니다.
async def reboot_handler(request):
    print("⚠️ [비상 경고] 설정 화면으로부터 원격 재부팅 명령을 수신했습니다!")
    subprocess.Popen(["sudo", "reboot"])
    return web.Response(text="REBOOTING")

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
# WebRTC 시그널링 엔드포인트
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

def get_current_wifi():
    try:
        cmd = "nmcli -t -f active,ssid dev wifi | egrep '^yes'"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        if ":" in output:
            return output.split(":")[1]
        return "Disconnected"
    except Exception:
        return "Wi-Fi"

async def get_status(request):
    global shared_status_cache
    shared_status_cache["wifi"] = get_current_wifi()
    
    # 🌟 [개인화 추가] 문지기가 request에 얹어준 현재 세션 유저 아이디를 캐시에 실어 보냅니다.
    shared_status_cache["current_user"] = request.get('user', 'Guest')
    
    return web.json_response(shared_status_cache)

# ==========================================
# 서버 가동 및 비상 안전 장치 구역
# ==========================================
async def start_background_tasks(app_context):
    sensor_driver.start_loop() 
    print("🌡️ [3단계] 실시간 체온 센서 하드웨어 루프 가동 완료")

    current_loop = asyncio.get_running_loop()
    t_cam = threading.Thread(target=camera_reader_thread, args=(current_loop,), daemon=True)
    t_cam.start()

    app_context['camera_loop'] = asyncio.create_task(camera_inference_loop())
    app_context['cleaner_loop'] = asyncio.create_task(memory_leak_cleaner_loop())

async def cleanup_background_tasks(app_context):
    app_context['camera_loop'].cancel()
    app_context['cleaner_loop'].cancel()
    await asyncio.gather(app_context['camera_loop'], app_context['cleaner_loop'], return_exceptions=True)

async def on_shutdown(app_context):
    global is_running
    print("🚨 로봇 관제 웹 서버 정지 절차에 진입합니다.")
    is_running = False  
    time.sleep(0.2)
    try:
        set_motor_speed(0, 0)
    except:
        pass
    cap.release()
    sensor_driver.close() 
    
    close_connections = [pc.close() for pc in pcs]
    if close_connections:
        await asyncio.gather(*close_connections)
    pcs.clear()
    print("👋 자원 및 SPI 통신 자원 반납 정상 완료")

# 🌟 login.py의 철통 문지기 미들웨어를 사용하도록 지정합니다.
app = web.Application(middlewares=[auth_middleware])
app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)

# 🌐 웹 페이지 서빙 라우팅 리스트 정리
app.router.add_get("/", index)
app.router.add_get("/monitor", monitor)
app.router.add_get("/control_page", control_page)
app.router.add_get("/settings", settings) 
app.router.add_get("/style.css", style)

# ⚡ 실시간 통신 및 기능 API 엔드포인트 매핑
app.router.add_post("/offer", offer)
app.router.add_post("/control", control)
app.router.add_get("/status", get_status)

# 🔑 login.py의 데이터 핸들러 함수들을 매핑합니다 (POST 보안 철저)
app.router.add_post("/login", login_handler)
app.router.add_post("/register", register_handler)
app.router.add_post("/logout", logout_handler) 
app.router.add_post("/reboot", reboot_handler) 

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
            sensor_driver.close()
        except:
            pass
        print("👋 시스템 완전히 안전 종료됨.")