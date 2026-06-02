from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
from ultralytics import YOLO
import cv2
import os
import time
import torch
import itertools
import math
from pathlib import Path
import pymysql
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from threading import Lock

app = Flask(__name__)
app.secret_key = "bike_detection_secret_key_2026"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "bike_detection",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

MODEL_OPTIONS = {
    "best": {
        "path": Path("best.pt"),
        "name": "best.pt 综合检测模型",
        "description": "可识别共享单车和其他载具，适合完整车流统计。"
    },
    "best_1": {
        "path": Path("best_1.pt"),
        "name": "best_1.pt 共享单车增强模型",
        "description": "重点提升共享单车识别能力，不用于其他载具统计。"
    }
}
ACTIVE_MODEL_KEY = os.environ.get("ACTIVE_MODEL_KEY", "best")
if ACTIVE_MODEL_KEY not in MODEL_OPTIONS:
    ACTIVE_MODEL_KEY = "best"
MODEL_PATH = MODEL_OPTIONS[ACTIVE_MODEL_KEY]["path"]
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMPUS_CAMERAS = {
    "A": {"name": "至臻楼", "camera_id": 0},
    "B": {"name": "友园15号楼", "camera_id": 0},
    "C": {"name": "食堂", "camera_id": 0},
    "D": {"name": "安楼", "camera_id": 0},
}
STATION_NAME_TO_CODE = {cfg["name"]: key for key, cfg in CAMPUS_CAMERAS.items()}
MODEL_TIME_PERIODS = {
    "morning": "早高峰",
    "noon": "午间",
    "evening": "晚高峰",
}
DISPATCH_DEPOT = "O"
DISPATCH_STATIONS = ["A", "B", "C", "D"]
DISPATCH_NODES = [DISPATCH_DEPOT] + DISPATCH_STATIONS
DISPATCH_Q = 15
DISPATCH_COORDS = {
    "O": (300.0, 377.0),
    "A": (0.0, 0.0),
    "B": (573.0, 712.0),
    "C": (806.0, 325.0),
    "D": (461.0, 0.0),
}
DISPATCH_DEMANDS = {
    "morning": {"A": 0, "B": 40, "C": 0, "D": 0},
    "noon": {"A": 30, "B": 0, "C": 0, "D": 30},
    "evening": {"A": 30, "B": 30, "C": 30, "D": 30},
}
DISPATCH_ALPHA_STATION = 1.0
DISPATCH_BETA_STATION = 0.2
DISPATCH_ALPHA_CENTER = 2.0
DISPATCH_BETA_CENTER = 1.0

campus_caps = {}
campus_latest_info = {}


class StaticImageCapture:
    def __init__(self, image_path, width=640, height=480):
        self.image_path = Path(image_path)
        image = cv2.imread(str(self.image_path))

        if image is None:
            raise FileNotFoundError(f"静态摄像头图片读取失败：{self.image_path}")

        self.frame = cv2.resize(image, (width, height))
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, *_args):
        return True

    def read(self):
        time.sleep(0.03)
        return True, self.frame.copy()


class VideoFileCapture:
    def __init__(self, video_path, width=640, height=480, loop=True):
        self.video_path = Path(video_path)
        self.width = width
        self.height = height
        self.loop = loop
        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise FileNotFoundError(f"视频流模拟文件读取失败：{self.video_path}")

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / fps if fps and fps > 1 else 1.0 / 30.0
        self.last_frame_time = 0.0

    def isOpened(self):
        return self.capture.isOpened()

    def set(self, *_args):
        return True

    def read(self):
        elapsed = time.time() - self.last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)

        ok, frame = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()

        self.last_frame_time = time.time()

        if not ok or frame is None:
            return False, None

        return True, cv2.resize(frame, (self.width, self.height))

    def release(self):
        self.capture.release()


def get_simulated_camera_source(stream_key):
    specific_video_key = f"STATIC_CAMERA_{stream_key.upper()}_VIDEO"
    specific_image_key = f"STATIC_CAMERA_{stream_key.upper()}_IMAGE"
    video_path = os.environ.get(specific_video_key)
    image_path = os.environ.get(specific_image_key)

    selected_key = os.environ.get("SIMULATED_CAMERA_KEY", "").strip().lower()
    selected_video = os.environ.get("SIMULATED_CAMERA_VIDEO", "").strip()
    selected_image = os.environ.get("SIMULATED_CAMERA_IMAGE", "").strip()

    if not video_path and selected_key == stream_key and selected_video:
        video_path = selected_video

    if not image_path and not video_path and selected_key == stream_key and selected_image:
        image_path = selected_image

    if video_path:
        return "video", video_path

    if image_path:
        return "image", image_path

    return None, None


def create_capture(stream_key, camera_id, width=640, height=480):
    source_type, source_path = get_simulated_camera_source(stream_key)

    if source_type == "video":
        print(f"{stream_key} 使用视频文件模拟摄像头：{source_path}")
        return VideoFileCapture(source_path, width, height)

    if source_type == "image":
        print(f"{stream_key} 使用静态图片模拟摄像头：{source_path}")
        return StaticImageCapture(source_path, width, height)

    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


for key, cfg in CAMPUS_CAMERAS.items():
    cap_i = create_capture(key, cfg["camera_id"], CAMERA_WIDTH, CAMERA_HEIGHT)

    campus_caps[key] = cap_i

    campus_latest_info[key] = {
        "name": cfg["name"],
        "shared_bike_count": 0,
        "other_vehicle_count": 0,
        "total_vehicle_count": 0,
        "congestion": "不拥堵",
        "fps": 0,
        "camera_status": "正常" if cap_i.isOpened() else "打开失败",
        "tracker": "ByteTrack",
        "active_track_count": 0,
        "unique_track_count": 0,
        "model_key": active_model_key if "active_model_key" in globals() else ACTIVE_MODEL_KEY,
        "model_name": MODEL_OPTIONS[ACTIVE_MODEL_KEY]["name"],
        "model_description": MODEL_OPTIONS[ACTIVE_MODEL_KEY]["description"]
    }

IMG_SIZE = 1280
CONF_THRES = 0.35
IOU_THRES = 0.45
TRACKER_CONFIG = "bytetrack.yaml"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = True if DEVICE != "cpu" else False

SHARED_BIKE_CLASSES = ["shared_bike", "共享单车"]


def is_shared_bike_class(cls_name, model_key=None):
    if model_key == "best_1":
        return True

    cls_text = str(cls_name).strip().lower()
    return cls_name in SHARED_BIKE_CLASSES or cls_text in {"bike", "bicycle", "shared bike", "shared_bike"}


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return "权限不足：只有管理员可以访问该页面", 403
        return func(*args, **kwargs)
    return wrapper


def get_congestion_level(total_count):
    if total_count >= 7:
        return "严重拥堵"
    elif total_count >= 4:
        return "拥堵"
    else:
        return "不拥堵"


for model_key, model_cfg in MODEL_OPTIONS.items():
    if not model_cfg["path"].exists():
        raise FileNotFoundError(f"找不到模型文件：{model_cfg['path']}")

model_switch_lock = Lock()
active_model_key = ACTIVE_MODEL_KEY
model = YOLO(str(MODEL_OPTIONS[active_model_key]["path"]))
print("当前模型：", MODEL_OPTIONS[active_model_key]["name"])
print("模型类别：", model.names)

tracking_models = {"main": model}
tracking_model_keys = {"main": active_model_key}
tracking_locks = {"main": Lock()}
tracking_states = {"main": {"seen_ids": set()}}
traffic_counter_states = {}
traffic_counter_locks = {}


def create_traffic_counter_state():
    return {
        "running": False,
        "started_at": None,
        "counted_ids": set(),
        "shared_bike_count": 0,
        "other_vehicle_count": 0,
        "total_vehicle_count": 0
    }


def get_traffic_counter(stream_key):
    if stream_key not in traffic_counter_states:
        traffic_counter_states[stream_key] = create_traffic_counter_state()
        traffic_counter_locks[stream_key] = Lock()

    return traffic_counter_states[stream_key], traffic_counter_locks[stream_key]


def traffic_counter_snapshot(stream_key):
    state, state_lock = get_traffic_counter(stream_key)

    with state_lock:
        started_at = state["started_at"]
        duration_seconds = 0
        if started_at is not None:
            duration_seconds = int((datetime.now() - started_at).total_seconds())

        return {
            "running": state["running"],
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S") if started_at else "",
            "duration_seconds": duration_seconds,
            "shared_bike_count": state["shared_bike_count"],
            "other_vehicle_count": state["other_vehicle_count"],
            "total_vehicle_count": state["total_vehicle_count"]
        }


def start_traffic_counter(stream_key):
    state, state_lock = get_traffic_counter(stream_key)

    with state_lock:
        state["running"] = True
        state["started_at"] = datetime.now()
        state["counted_ids"] = set()
        state["shared_bike_count"] = 0
        state["other_vehicle_count"] = 0
        state["total_vehicle_count"] = 0

    return traffic_counter_snapshot(stream_key)


def stop_traffic_counter(stream_key):
    state, state_lock = get_traffic_counter(stream_key)

    with state_lock:
        state["running"] = False

    return traffic_counter_snapshot(stream_key)


def reset_traffic_counter(stream_key):
    state, state_lock = get_traffic_counter(stream_key)

    with state_lock:
        state.update(create_traffic_counter_state())

    return traffic_counter_snapshot(stream_key)


def count_tracked_vehicle(stream_key, object_key, cls_name, model_key=None):
    state, state_lock = get_traffic_counter(stream_key)

    with state_lock:
        if not state["running"] or object_key in state["counted_ids"]:
            return

        state["counted_ids"].add(object_key)

        if is_shared_bike_class(cls_name, model_key):
            state["shared_bike_count"] += 1
        else:
            state["other_vehicle_count"] += 1

        state["total_vehicle_count"] += 1


def get_active_model_info():
    cfg = MODEL_OPTIONS[active_model_key]
    return {
        "key": active_model_key,
        "name": cfg["name"],
        "path": str(cfg["path"]),
        "description": cfg["description"]
    }


def get_model_options_payload():
    return {
        key: {
            "key": key,
            "name": cfg["name"],
            "path": str(cfg["path"]),
            "description": cfg["description"]
        }
        for key, cfg in MODEL_OPTIONS.items()
    }


def reset_runtime_after_model_switch():
    tracking_models.clear()
    tracking_model_keys.clear()
    tracking_locks.clear()
    tracking_states.clear()
    traffic_counter_states.clear()
    traffic_counter_locks.clear()


def switch_active_model(model_key):
    global active_model_key, model, MODEL_PATH

    if model_key not in MODEL_OPTIONS:
        raise ValueError("未知模型权重")

    model_cfg = MODEL_OPTIONS[model_key]
    new_model = YOLO(str(model_cfg["path"]))

    with model_switch_lock:
        active_model_key = model_key
        MODEL_PATH = model_cfg["path"]
        model = new_model
        reset_runtime_after_model_switch()
        tracking_models["main"] = model
        tracking_model_keys["main"] = active_model_key
        tracking_locks["main"] = Lock()
        tracking_states["main"] = {"seen_ids": set()}

    print("已切换模型：", model_cfg["name"])
    print("模型类别：", model.names)
    return get_active_model_info()


def get_tracking_model(stream_key):
    with model_switch_lock:
        if stream_key not in tracking_models or tracking_model_keys.get(stream_key) != active_model_key:
            tracking_models[stream_key] = YOLO(str(MODEL_OPTIONS[active_model_key]["path"]))
            tracking_model_keys[stream_key] = active_model_key
            tracking_locks[stream_key] = Lock()
            tracking_states[stream_key] = {"seen_ids": set()}

        return tracking_models[stream_key], tracking_locks[stream_key], tracking_states[stream_key]


def detect_tracked_frame(frame, stream_key="main"):
    track_model, track_lock, track_state = get_tracking_model(stream_key)
    model_key = tracking_model_keys.get(stream_key, active_model_key)

    with track_lock:
        results = track_model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=DEVICE,
            half=HALF,
            verbose=False
        )

    result = results[0]
    annotated_frame = result.plot()

    shared_track_ids = set()
    other_track_ids = set()
    shared_fallback_count = 0
    other_fallback_count = 0
    tracked_objects = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = track_model.names[cls_id]
            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])
                object_key = (cls_id, track_id)
                track_state["seen_ids"].add(object_key)
                tracked_objects.append((object_key, cls_name))

            if is_shared_bike_class(cls_name, model_key):
                if track_id is None:
                    shared_fallback_count += 1
                else:
                    shared_track_ids.add(track_id)
            else:
                if track_id is None:
                    other_fallback_count += 1
                else:
                    other_track_ids.add(track_id)

    for object_key, cls_name in tracked_objects:
        count_tracked_vehicle(stream_key, object_key, cls_name, model_key)

    shared_bike_count = len(shared_track_ids) + shared_fallback_count
    other_vehicle_count = len(other_track_ids) + other_fallback_count
    total_vehicle_count = shared_bike_count + other_vehicle_count
    congestion = get_congestion_level(total_vehicle_count)
    active_track_count = len(shared_track_ids | other_track_ids)
    unique_track_count = len(track_state["seen_ids"])

    return (
        annotated_frame,
        shared_bike_count,
        other_vehicle_count,
        total_vehicle_count,
        congestion,
        active_track_count,
        unique_track_count
    )

cap = create_capture("main", CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT)

if not cap.isOpened():
    print("主摄像头打开失败：如需演示，可使用 run_with_static_image.py 指定静态图片模拟。")
latest_info = {
    "fps": 0,
    "shared_bike_count": 0,
    "other_vehicle_count": 0,
    "total_vehicle_count": 0,
    "congestion": "不拥堵",
    "device": "CUDA GPU" if DEVICE != "cpu" else "CPU",
    "camera_status": "正常",
    "tracker": "ByteTrack",
    "active_track_count": 0,
    "unique_track_count": 0,
    "model_key": active_model_key,
    "model_name": MODEL_OPTIONS[active_model_key]["name"],
    "model_description": MODEL_OPTIONS[active_model_key]["description"],
    "traffic_counter_running": False,
    "traffic_counter_started_at": "",
    "traffic_counter_duration_seconds": 0,
    "traffic_counter_shared_bike_count": 0,
    "traffic_counter_other_vehicle_count": 0,
    "traffic_counter_total_vehicle_count": 0
}


@app.route("/init_admin")
def init_admin():
    """
    第一次运行时访问：
    http://127.0.0.1:5000/init_admin

    默认管理员：
    用户名：admin
    密码：admin123
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username=%s", ("admin",))
    exist = cursor.fetchone()

    if exist:
        cursor.close()
        conn.close()
        return "管理员账号已存在。"

    password_hash = generate_password_hash("admin123")

    cursor.execute(
        """
        INSERT INTO users(username, password_hash, role, create_time)
        VALUES (%s, %s, %s, %s)
        """,
        ("admin", password_hash, "admin", datetime.now())
    )

    conn.commit()
    cursor.close()
    conn.close()

    return "管理员初始化成功：用户名 admin，密码 admin123。请登录后及时修改。"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("login.html", error="用户名和密码不能为空")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user is None:
        return render_template("login.html", error="用户不存在")

    if not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="密码错误")

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]

    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if not username or not password:
        return render_template("register.html", error="用户名和密码不能为空")

    if password != confirm_password:
        return render_template("register.html", error="两次输入的密码不一致")

    password_hash = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users(username, password_hash, role, create_time)
            VALUES (%s, %s, %s, %s)
            """,
            (username, password_hash, "user", datetime.now())
        )

        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for("login"))

    except pymysql.err.IntegrityError:
        return render_template("register.html", error="用户名已存在")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        username=session.get("username"),
        role=session.get("role")
    )


def generate_frames():
    global latest_info

    prev_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            latest_info["camera_status"] = "读取失败"
            continue

        (
            annotated_frame,
            shared_bike_count,
            other_vehicle_count,
            total_vehicle_count,
            congestion,
            active_track_count,
            unique_track_count
        ) = detect_tracked_frame(frame, "main")

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        traffic_counter = traffic_counter_snapshot("main")
        active_model = get_active_model_info()

        latest_info = {
            "fps": round(fps, 1),
            "shared_bike_count": shared_bike_count,
            "other_vehicle_count": other_vehicle_count,
            "total_vehicle_count": total_vehicle_count,
            "congestion": congestion,
            "device": "CUDA GPU" if DEVICE != "cpu" else "CPU",
            "camera_status": "正常",
            "tracker": "ByteTrack",
            "active_track_count": active_track_count,
            "unique_track_count": unique_track_count,
            "model_key": active_model["key"],
            "model_name": active_model["name"],
            "model_description": active_model["description"],
            "traffic_counter_running": traffic_counter["running"],
            "traffic_counter_started_at": traffic_counter["started_at"],
            "traffic_counter_duration_seconds": traffic_counter["duration_seconds"],
            "traffic_counter_shared_bike_count": traffic_counter["shared_bike_count"],
            "traffic_counter_other_vehicle_count": traffic_counter["other_vehicle_count"],
            "traffic_counter_total_vehicle_count": traffic_counter["total_vehicle_count"]
        }

        english_congestion_map = {
            "不拥堵": "Normal",
            "拥堵": "Crowded",
            "严重拥堵": "Severe"
        }

        cv2.putText(
            annotated_frame,
            "Any Location Monitoring",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Shared bikes: {shared_bike_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Other vehicles: {other_vehicle_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Total: {total_vehicle_count}  Status: {english_congestion_map.get(congestion, 'Normal')}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Tracker: ByteTrack  Active IDs: {active_track_count}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        counter_status = "ON" if traffic_counter["running"] else "OFF"
        cv2.putText(
            annotated_frame,
            f"Flow counter: {counter_status}  Passed total: {traffic_counter['total_vehicle_count']}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", annotated_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

def detect_one_frame(frame, stream_key):
    return detect_tracked_frame(frame, stream_key)
def generate_campus_frames(camera_key):
    cap_i = campus_caps[camera_key]
    prev_time = time.time()

    english_name_map = {
        "A": "Zhizhen Building",
        "B": "Youyuan 15",
        "C": "Canteen",
        "D": "An Building",
    }

    english_congestion_map = {
        "不拥堵": "Normal",
        "拥堵": "Crowded",
        "严重拥堵": "Severe"
    }

    while True:
        success, frame = cap_i.read()

        if not success:
            campus_latest_info[camera_key]["camera_status"] = "读取失败"
            continue

        (
            annotated_frame,
            shared_count,
            other_count,
            total_count,
            congestion,
            active_track_count,
            unique_track_count
        ) = detect_one_frame(frame, camera_key)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        active_model = get_active_model_info()

        campus_latest_info[camera_key] = {
            "name": CAMPUS_CAMERAS[camera_key]["name"],
            "shared_bike_count": shared_count,
            "other_vehicle_count": other_count,
            "total_vehicle_count": total_count,
            "congestion": congestion,
            "fps": round(fps, 1),
            "camera_status": "正常",
            "tracker": "ByteTrack",
            "active_track_count": active_track_count,
            "unique_track_count": unique_track_count,
            "model_key": active_model["key"],
            "model_name": active_model["name"],
            "model_description": active_model["description"]
        }

        # 注意：这里画面上只写英文，避免 OpenCV 中文乱码
        location_text = english_name_map.get(camera_key, camera_key)
        congestion_text = english_congestion_map.get(congestion, "Normal")

        cv2.putText(
            annotated_frame,
            location_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Shared bikes: {shared_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Other vehicles: {other_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Total: {total_count}  Status: {congestion_text}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Tracker: ByteTrack  Active IDs: {active_track_count}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", annotated_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


def get_latest_station_counts_from_records():
    counts = {}
    source_rows = {}

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for station, cfg in CAMPUS_CAMERAS.items():
            cursor.execute(
                """
                SELECT location, shared_bike_count, total_vehicle_count, record_time
                FROM campus_flow_records
                WHERE location = %s
                ORDER BY record_time DESC
                LIMIT 1
                """,
                (cfg["name"],)
            )
            row = cursor.fetchone()

            if row:
                counts[station] = int(row["shared_bike_count"] or 0)
                if row.get("record_time") is not None:
                    row["record_time"] = str(row["record_time"])
                source_rows[station] = row
            else:
                counts[station] = int(campus_latest_info.get(station, {}).get("shared_bike_count", 0))
                source_rows[station] = {
                    "location": cfg["name"],
                    "shared_bike_count": counts[station],
                    "total_vehicle_count": campus_latest_info.get(station, {}).get("total_vehicle_count", 0),
                    "record_time": "实时画面，尚未保存"
                }
    finally:
        cursor.close()
        conn.close()

    return counts, source_rows


def normalize_dispatch_result(result):
    node_names = {"O": "配送中心"}
    node_names.update({station: cfg["name"] for station, cfg in CAMPUS_CAMERAS.items()})

    for step in result["steps"]:
        step["from_name"] = node_names.get(step["from"], step["from_name"])
        step["to_name"] = node_names.get(step["to"], step["to_name"])

    result["node_names"] = node_names
    return result


def dispatch_distance(i, j):
    xi, yi = DISPATCH_COORDS[i]
    xj, yj = DISPATCH_COORDS[j]
    return math.hypot(xi - xj, yi - yj)


def dispatch_alpha(node):
    return DISPATCH_ALPHA_CENTER if node == DISPATCH_DEPOT else DISPATCH_ALPHA_STATION


def dispatch_beta(node):
    return DISPATCH_BETA_CENTER if node == DISPATCH_DEPOT else DISPATCH_BETA_STATION


def generate_dispatch_routes(max_legs):
    routes = []

    for legs in range(2, max_legs + 1):
        middle_len = legs - 1

        for middle in itertools.product(DISPATCH_STATIONS, repeat=middle_len):
            route = [DISPATCH_DEPOT] + list(middle) + [DISPATCH_DEPOT]

            if all(a != b for a, b in zip(route[:-1], route[1:])):
                routes.append(route)

    return routes


def evaluate_dispatch_route(route, current_counts, demands, center_supply_allowed):
    station_index = {station: idx for idx, station in enumerate(DISPATCH_STATIONS)}
    initial_state = tuple(int(current_counts[station]) for station in DISPATCH_STATIONS)
    states = {initial_state: (0.0, [])}
    legs = list(zip(route[:-1], route[1:]))

    for i, j in legs:
        next_states = {}
        distance = dispatch_distance(i, j)
        vehicle_cost = dispatch_alpha(i) * distance

        for state, (cost, carried) in states.items():
            if j == DISPATCH_DEPOT:
                q_values = [0]
            elif i == DISPATCH_DEPOT and not center_supply_allowed:
                q_values = [0]
            elif i == DISPATCH_DEPOT:
                q_values = range(0, DISPATCH_Q + 1)
            else:
                q_values = range(0, min(DISPATCH_Q, state[station_index[i]]) + 1)

            for q in q_values:
                new_state = list(state)

                if i in station_index:
                    new_state[station_index[i]] -= q

                if j in station_index:
                    new_state[station_index[j]] += q

                new_state = tuple(new_state)
                new_cost = cost + vehicle_cost + dispatch_beta(i) * q

                if new_state not in next_states or new_cost < next_states[new_state][0]:
                    next_states[new_state] = (new_cost, carried + [q])

        states = next_states

        if not states:
            return None

    best_state = None
    best_cost = None
    best_carried = None

    for state, (cost, carried) in states.items():
        feasible = all(
            state[station_index[station]] >= demands[station]
            for station in DISPATCH_STATIONS
        )

        if feasible and (best_cost is None or cost < best_cost):
            best_state = state
            best_cost = cost
            best_carried = carried

    if best_state is None:
        return None

    steps = []
    for step_index, ((i, j), bikes) in enumerate(zip(legs, best_carried), start=1):
        distance = dispatch_distance(i, j)
        vehicle_cost = dispatch_alpha(i) * distance
        bike_cost = dispatch_beta(i) * bikes
        steps.append({
            "step": step_index,
            "from": i,
            "to": j,
            "from_name": i,
            "to_name": j,
            "bikes": bikes,
            "distance": distance,
            "vehicle_cost": vehicle_cost,
            "bike_cost": bike_cost,
            "total_cost": vehicle_cost + bike_cost,
        })

    final_counts = {
        station: int(best_state[station_index[station]])
        for station in DISPATCH_STATIONS
    }

    return {
        "route": route,
        "steps": steps,
        "final_counts": final_counts,
        "objective": best_cost,
        "vehicle_cost": sum(step["vehicle_cost"] for step in steps),
        "bike_cost": sum(step["bike_cost"] for step in steps),
    }


def solve_dispatch_fallback(current_counts, time_period, max_legs=6):
    demands = DISPATCH_DEMANDS[time_period]
    total_current = sum(current_counts[station] for station in DISPATCH_STATIONS)
    total_required = sum(demands[station] for station in DISPATCH_STATIONS)
    center_supply_allowed = total_current < total_required

    if center_supply_allowed and total_current + DISPATCH_Q < total_required:
        raise RuntimeError(
            f"当前记录总车数为 {total_current}，{MODEL_TIME_PERIODS.get(time_period, time_period)}最低需求为 {total_required}。"
            f"模型每次从配送中心最多补 {DISPATCH_Q} 辆，当前数据下无可行路线。"
        )

    routes = generate_dispatch_routes(max_legs)
    best = None
    feasible_count = 0

    for route in routes:
        result = evaluate_dispatch_route(
            route=route,
            current_counts=current_counts,
            demands=demands,
            center_supply_allowed=center_supply_allowed
        )

        if result is None:
            continue

        feasible_count += 1

        if best is None or result["objective"] < best["objective"]:
            best = result

    if best is None:
        raise RuntimeError(f"未找到可行调度路线，请增加最大路线段数或检查站点车辆数量。")

    best["time_period"] = time_period
    best["current_counts"] = current_counts
    best["demands"] = demands
    best["total_current"] = total_current
    best["total_required"] = total_required
    best["center_supply_allowed"] = center_supply_allowed
    best["feasible_route_count"] = feasible_count
    best["candidate_route_count"] = len(routes)
    best["solver_backend"] = "内置动态规划求解器"
    return best


def solve_dispatch_from_records(time_period="evening", max_legs=6):
    current_counts, source_rows = get_latest_station_counts_from_records()

    try:
        import 建模 as dispatch_model
    except Exception as exc:
        result = solve_dispatch_fallback(
            current_counts=current_counts,
            time_period=time_period,
            max_legs=max_legs
        )
        result["solver_warning"] = f"未使用 Gurobi，原因：{exc}"
    else:
        result = dispatch_model.solve_complete_plan(
            current_counts=current_counts,
            time_period=time_period,
            max_legs=max_legs
        )
        result["solver_backend"] = "建模.py / Gurobi"

    result = normalize_dispatch_result(result)
    result["source_rows"] = source_rows
    result["time_period_name"] = MODEL_TIME_PERIODS.get(time_period, time_period)
    return result


@app.route("/campus_flow")
@login_required
def campus_flow():
    return render_template(
        "campus_flow.html",
        username=session.get("username"),
        role=session.get("role")
    )


@app.route("/campus_video_feed/<camera_key>")
@login_required
def campus_video_feed(camera_key):
    if camera_key not in CAMPUS_CAMERAS:
        return "摄像头不存在", 404

    return Response(
        generate_campus_frames(camera_key),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/campus_info")
@login_required
def campus_info():
    return jsonify(campus_latest_info)


@app.route("/dispatch_model/solve", methods=["POST"])
@login_required
def dispatch_model_solve():
    data = request.get_json(silent=True) or {}
    time_period = data.get("time_period", "evening")
    max_legs = int(data.get("max_legs", 6))

    if time_period not in MODEL_TIME_PERIODS:
        return jsonify({
            "success": False,
            "message": "未知时间段，请选择 morning、noon 或 evening。"
        })

    try:
        result = solve_dispatch_from_records(time_period=time_period, max_legs=max_legs)
        return jsonify({
            "success": True,
            "message": "调度模型求解完成",
            "result": result
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "message": str(exc)
        })
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detect_info")
@login_required
def detect_info():
    return jsonify(latest_info)


@app.route("/model_options")
@login_required
def model_options():
    return jsonify({
        "success": True,
        "active_model": get_active_model_info(),
        "models": get_model_options_payload()
    })


@app.route("/switch_model", methods=["POST"])
@login_required
def switch_model():
    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key", "").strip()

    try:
        active_model = switch_active_model(model_key)
        return jsonify({
            "success": True,
            "message": f"已切换为 {active_model['name']}，跟踪状态和车流累计已重置。",
            "active_model": active_model,
            "models": get_model_options_payload()
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "message": f"模型切换失败：{exc}",
            "active_model": get_active_model_info(),
            "models": get_model_options_payload()
        })


@app.route("/traffic_counter/start", methods=["POST"])
@login_required
def start_main_traffic_counter():
    return jsonify({
        "success": True,
        "message": "车流量统计已开始",
        "counter": start_traffic_counter("main")
    })


@app.route("/traffic_counter/stop", methods=["POST"])
@login_required
def stop_main_traffic_counter():
    return jsonify({
        "success": True,
        "message": "车流量统计已停止",
        "counter": stop_traffic_counter("main")
    })


@app.route("/traffic_counter/reset", methods=["POST"])
@login_required
def reset_main_traffic_counter():
    return jsonify({
        "success": True,
        "message": "车流量统计已清零",
        "counter": reset_traffic_counter("main")
    })


@app.route("/save_record", methods=["POST"])
@login_required
def save_record():
    data = request.get_json()
    scene = data.get("scene", "").strip()

    if not scene:
        return jsonify({"success": False, "message": "场景不能为空"})

    record_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO detection_records
        (scene, shared_bike_count, other_vehicle_count, total_vehicle_count, congestion, record_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(sql, (
            scene,
            latest_info["shared_bike_count"],
            latest_info["other_vehicle_count"],
            latest_info["total_vehicle_count"],
            latest_info["congestion"],
            record_time
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "记录保存成功",
            "record": {
                "scene": scene,
                "shared_bike_count": latest_info["shared_bike_count"],
                "other_vehicle_count": latest_info["other_vehicle_count"],
                "total_vehicle_count": latest_info["total_vehicle_count"],
                "congestion": latest_info["congestion"],
                "record_time": record_time
            }
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"数据库保存失败：{str(e)}"})


@app.route("/admin/records")
@admin_required
def admin_records():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 任意地点检测记录
    cursor.execute("""
        SELECT *
        FROM detection_records
        ORDER BY record_time DESC
        LIMIT 200
    """)
    location_records = cursor.fetchall()

    # 校园流动检测记录
    cursor.execute("""
        SELECT *
        FROM campus_flow_records
        ORDER BY record_time DESC
        LIMIT 300
    """)
    flow_records = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "admin_records.html",
        location_records=location_records,
        flow_records=flow_records,
        username=session.get("username"),
        role=session.get("role")
    )

@app.route("/save_campus_flow_record", methods=["POST"])
@login_required
def save_campus_flow_record():
    record_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO campus_flow_records
        (location, shared_bike_count, other_vehicle_count, total_vehicle_count, congestion, record_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        for key, info in campus_latest_info.items():
            cursor.execute(sql, (
                info["name"],
                info["shared_bike_count"],
                info["other_vehicle_count"],
                info["total_vehicle_count"],
                info["congestion"],
                record_time
            ))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "校园流动检测数据保存成功",
            "record_time": record_time
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"数据库保存失败：{str(e)}"
        })
@app.route("/admin/delete_record/<int:record_id>", methods=["POST"])
@admin_required
def delete_record(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM detection_records WHERE id=%s", (record_id,))
    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for("admin_records"))


if __name__ == "__main__":
    print("=" * 60)
    print("校园共享单车检测系统启动")
    print("CUDA 是否可用：", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 GPU：", torch.cuda.get_device_name(0))
    print("访问地址：http://127.0.0.1:5000")
    print("登录地址：http://127.0.0.1:5000/login")
    print("初始化管理员：http://127.0.0.1:5000/init_admin")
    print("=" * 60)

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
