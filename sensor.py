import time
import random
import threading

class TemperatureSensor:
    def __init__(self):
        self.enabled = True
        self.pet_temp = 36.5
        self.is_running = True
        
        try:
            import os
            os.environ.setdefault("JETSON_MODEL_NAME", "JETSON_ORIN_NANO")
            import spidev
            import Jetson.GPIO as GPIO

            self.BUS, self.CS_DEV = 0, 0
            self.CS_BCM = 8
            self.SPI_MODE = 3
            self.SPI_SPEED = 1_000_000
            self.CMD_OBJ = 0xA0
            self.CMD_SEN = 0xA1

            self.spi = spidev.SpiDev()
            self.spi.open(self.BUS, self.CS_DEV)
            self.spi.mode = self.SPI_MODE
            self.spi.max_speed_hz = self.SPI_SPEED

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.CS_BCM, GPIO.OUT, initial=GPIO.HIGH)
            
            # 초기 안정화 센서 예열
            for _ in range(3):
                try:
                    self._read16(self.CMD_OBJ)
                    self._read16(self.CMD_SEN)
                except:
                    pass
                time.sleep(0.05)
                
            print(" Peterson 실물 DTPM151 하드웨어 센서 초기화 완료")
        except Exception as e:
            print(f"⚠️ [하드웨어 알림] 젯슨 SPI 센서를 초기화할 수 없습니다 ({e}). 시뮬레이션 모드로 전환합니다.")
            self.enabled = False

    def _usleep(self, us):
        time.sleep(us / 1_000_000.0)

    def _s16(self, v):
        return v - 0x10000 if (v & 0x8000) else v

    def _read16(self, cmd):
        import Jetson.GPIO as GPIO
        if not self.enabled:
            return 0
        GPIO.output(self.CS_BCM, GPIO.LOW)
        self._usleep(10)
        self.spi.xfer2([cmd])
        self._usleep(10)
        lo = self.spi.xfer2([0x22])[0]
        self._usleep(10)
        hi = self.spi.xfer2([0x22])[0]
        self._usleep(10)
        GPIO.output(self.CS_BCM, GPIO.HIGH)
        return (hi << 8) | lo

    def read_object_c(self):
        return round(self._s16(self._read16(self.CMD_OBJ)) / 10.0, 1)

    def read_sensor_c(self):
        return round(self._s16(self._read16(self.CMD_SEN)) / 10.0, 1)

    def start_loop(self):
        """백그라운드에서 온도를 주기적으로 계측하는 스레드 시작"""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        if not self.enabled:
            print("🌡️ 가상 체온 시뮬레이터 가동 시작")
            while self.is_running:
                self.pet_temp = round(36.5 + random.uniform(-0.2, 0.2), 1)
                time.sleep(0.5)
            return

        print(" An 실물 DTPM151 하드웨어 센서 루프 가동 시작")
        while self.is_running:
            try:
                tobj = self.read_object_c()
                tsen = self.read_sensor_c()
                print(f"📡 [RAW 디버깅] 체온(Obj): {tobj:.1f} °C | 센서주변(Sen): {tsen:.1f} °C")
                
                if 15.0 < tobj < 45.0:  
                    self.pet_temp = tobj
            except Exception as e:
                print(f"⚠️ 센서 읽기 실패 원인: {e}")
            time.sleep(0.5)

    def close(self):
        self.is_running = False
        if self.enabled:
            try:
                self.spi.close()
            except:
                pass