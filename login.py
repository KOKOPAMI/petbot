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

PUBLIC_GET_PATHS = {"/login", "/style.css"}
PUBLIC_POST_PATHS = {"/login", "/register", "/logout"}

# 🛡️ [다중 사용자용 문지기] 단순 true 검사가 아닌, 어떤 사용자인지 아이디를 식별합니다.
@web.middleware
async def auth_middleware(request, handler):
    if request.path in PUBLIC_GET_PATHS:
        return await handler(request)
    if request.method == "POST" and request.path in PUBLIC_POST_PATHS:
        return await handler(request)

    session_user = request.cookies.get("session_user")
    if not session_user:
        raise web.HTTPFound("/login")

    request['user'] = session_user
    return await handler(request)


async def login_page_handler(request):
    with open("login.html", "r", encoding="utf-8") as f:
        return web.Response(
            text=f.read(),
            content_type="text/html"
        )
        
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
        response.set_cookie(
            "session_user",
            username,
            httponly=True
        )
        return response
    else:
        return web.HTTPUnauthorized(text="FAIL")