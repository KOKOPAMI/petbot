import sqlite3
import hashlib
from aiohttp import web

DB_FILE = "petbot.db"

# 데이터베이스 초기화 및 테이블 생성
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# 비밀번호 암호화 함수
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# 🛡️ [다중 사용자용 문지기] 단순 true 검사가 아닌, 어떤 사용자인지 아이디를 식별합니다.
@web.middleware
async def auth_middleware(request, handler):
    # 로그인 폼, 회원가입 API, 스타일시트는 무조건 프리패스
    if request.path in ["/login", "/register", "/style.css"]:
        return await handler(request)
        
    # 🔑 쿠키에서 로그인한 사람의 '고유 아이디 명찰'을 꺼내옵니다.
    session_user = request.cookies.get("session_user")
    
    # 아이디 명찰이 없다면 로그인 안 한 사람이므로 로그인 창으로 튕겨냅니다.
    if not session_user:
        try:
            with open("login.html", "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type="text/html")
        except FileNotFoundError:
            return web.Response(status=404, text="login.html 파일을 찾을 수 없습니다.")
    
    # 💡 다른 페이지 핸들러에서 request['user']로 현재 로그인한 유저를 식별할 수 있게 배달
    request['user'] = session_user
    return await handler(request)

# 🔐 회원가입 API 처리
async def register_handler(request):
    data = await request.json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return web.Response(status=400, text="BAD_REQUEST")

    hashed = hash_password(password)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return web.Response(text="SUCCESS")
    except sqlite3.IntegrityError:
        return web.Response(status=400, text="EXISTS")
    finally:
        conn.close()

# 🔑 [다중 사용자용 로그인] 각자 고유한 아이디로 명찰 쿠키를 발급합니다.
async def login_handler(request):
    data = await request.json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    hashed = hash_password(password)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0] == hashed:
        response = web.Response(text="OK")
        # 🌟 실제 로그인한 유저의 "아이디"를 30일짜리 쿠키 명찰로 구워줍니다.
        response.set_cookie("session_user", username, max_age=2592000, httponly=True)
        return response
    else:
        return web.HTTPUnauthorized(text="FAIL")