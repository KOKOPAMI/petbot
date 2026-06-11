# main.py
import sys
import os
import asyncio
from aiohttp import web

# webrtc_server에서 완성된 랩핑 앱(app)과 종료 절차(on_shutdown)를 가져옵니다.
from webrtc_server import app, on_shutdown

def main():
    print("🚀 [Main Launcher] petbot 통합 인공지능 자율주행 시스템 구동을 시작합니다.")
    
    # SSL 보안 컨텍스트 설정 세팅 (아이폰 마이크 권한 획득용)
    import ssl
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    cert_path = 'cert.pem'
    key_path = 'key.pem'
    
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        print(f"❌ 에러: {cert_path} 또는 {key_path} 인증서 파일이 프로젝트 폴더에 없습니다!")
        print("💡 HTTPS 통신을 위해 인증서 파일(cert.pem, key.pem)을 먼저 생성해 주세요.")
        sys.exit(1)
        
    ssl_context.load_cert_chain(cert_path, key_path)

    try:
        # webrtc_server.py에 선언된 정석 라우터 세트를 기반으로 웹 앱 구동
        web.run_app(app, host="0.0.0.0", port=5000, ssl_context=ssl_context)
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 메인 런처 제어가 중단되었습니다 (Ctrl+C).")
    finally:
        print("🚨 [비상 안전 시스템] 프로세스 종료가 감지되어 최종 자원 반납 절차를 강제 실행합니다.")
        # 비동기 이벤트 루프를 얻어와 서버 종료(on_shutdown) 안전 장치를 마지막으로 작동시킵니다.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.run_until_complete(on_shutdown(app))
        except Exception as e:
            print(f"⚠️ 안전 종료 중 사소한 예외 발생 (무시 가능): {e}")
        print("👋 petbot 시스템이 안전하게 완전히 종료되었습니다.")

if __name__ == "__main__":
    main()