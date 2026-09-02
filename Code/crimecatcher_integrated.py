from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
import os, cv2, datetime, time, threading, pyttsx3, torch, json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not available. System health monitoring will be limited.")
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Full
from collections import deque, defaultdict

# === Detection modules ===
from send_sms import sendsms
from ultralytics import YOLO
from Fight_Detection_Pip_Package.fight_detection.Fight_utils import loadModel, fight_detect_on_frame

# ===== Config =====
WEAPON_MODEL_PATH = "models/best.pt"
FIGHT_MODEL_PATH = "models/model_ucf.pth"
CAMERAS = [
    {"id": "IPCam2", "source": "http://192.168.1.2:8080/video", "location": "No.14, Raghuvanahalli, Kanakapura Road, Bengaluru - 560109"},
    {"id": "IPCam1", "source": "http://192.168.1.4:8080/video", "location": "No.14, Raghuvanahalli, Kanakapura Road, Bengaluru - 560109"}
]
SMS_TO = "+918147087263"
TWILIO_FROM = "+12313034458"

STATUSES = {cam["id"]: "Monitoring" for cam in CAMERAS}
ALERTS_SENT = {cam["id"]: False for cam in CAMERAS}
PENDING_ALERTS = {cam["id"]: False for cam in CAMERAS}
SAVED_CLIPS_DIR = "saved_clips"
os.makedirs(SAVED_CLIPS_DIR, exist_ok=True)
VIDEO_CLIP_LENGTH = 60
VIDEO_BUFFERS = {cam["id"]: [] for cam in CAMERAS}
LAST_ALERT_TIME = {cam["id"]: 0 for cam in CAMERAS}
LAST_FRAME_TIME = {cam["id"]: time.time() for cam in CAMERAS}
OFFLINE_ALERTED = {cam["id"]: False for cam in CAMERAS}
COOLDOWN_SECONDS = 5
# camera tuning knobs to reduce lag while keeping fidelity
# you can tweak these down further if stream is still laggy
LOCAL_CAMERA_WIDTH = 480      # was 640
LOCAL_CAMERA_HEIGHT = 360     # was 480
LOCAL_CAMERA_FPS = 12         # was 18
LOCAL_CAMERA_BUFFERSIZE = 1
SMALL_FRAME_SIZE = (256, 192)  # base frame for lightweight operations
STREAM_FRAME_SIZE = (320, 240)  # was (416, 312), lower for smoother streaming
FIGHT_SEQUENCE_LENGTH = 16

# detection scheduling
# run heavy detection a bit less frequently to cut CPU/GPU usage
DETECTION_FRAME_GAP = 30        # was 20
DETECTION_FRAME_GAP_IP = 70    # IP cameras: much less frequent for smoother streaming
HIGH_RES_DETECTION_GAP = 3
IP_CAMERA_FRAME_SKIP = 3       # process every 3rd frame for IP cameras (more aggressive)
HIGH_RES_FRAME_SIZE = (512, 384)
IP_CAMERA_STREAM_SIZE = (256, 192)  # smaller stream size for IP cameras
IP_CAMERA_DETECTION_SIZE = (256, 192)  # smaller detection size for IP cameras
DETECTION_CONF_THRESHOLD = 0.5  # balanced: reduce false positives but still detect real knives
DETECTION_IOU_THRESHOLD = 0.45
VOICE_MAX_QUEUE = 5
MIN_ALERT_CONFIDENCE = 0.45  # smoothed confidence threshold for alerts
CONF_HISTORY = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# filter out unlikely boxes by size (relative to frame area)
MIN_BOX_AREA_RATIO = 0.005  # ignore very tiny boxes (e.g., fan edges, noise)
MAX_BOX_AREA_RATIO = 0.7    # ignore huge boxes covering most of the frame
# Additional filters to reduce false positives (fans, etc.)
MIN_ASPECT_RATIO = 0.3      # knives are usually taller than wide (aspect ratio > 0.3)
MAX_ASPECT_RATIO = 3.0      # knives are not extremely long (aspect ratio < 3.0)
TOP_FRAME_EXCLUDE = 0.15     # exclude top 15% of frame (where fans/ceiling objects often are)

weapon_model = YOLO(WEAPON_MODEL_PATH)
try:
    weapon_model.to(DEVICE)
    print(f"[MODEL] Weapon model moved to {DEVICE}")
except Exception as device_err:
    print(f"[MODEL] Could not move weapon model to {DEVICE}: {device_err}")
fight_model = loadModel(FIGHT_MODEL_PATH)
CLASS_NAMES = {0: "Gun", 1: "Knife"}

# require multiple frames before crime
WEAPON_COUNT = {cam["id"]: 0 for cam in CAMERAS}
WEAPON_CONF_HISTORY = {cam["id"]: deque(maxlen=CONF_HISTORY) for cam in CAMERAS}
FRAME_TIMES = {cam["id"]: deque(maxlen=30) for cam in CAMERAS}

# Analytics and tracking
ALERT_HISTORY = []  # Store all alerts with timestamps
DETECTION_STATS = {
    "total_detections": 0,
    "weapon_detections": defaultdict(int),
    "fight_detections": 0,
    "false_positives": 0,
    "alerts_sent": 0,
    "detections_by_hour": defaultdict(int),
    "detections_by_camera": defaultdict(int)
}
SYSTEM_START_TIME = time.time()

voice_queue = Queue(maxsize=VOICE_MAX_QUEUE)
weapon_lock = threading.Lock()
DETECTION_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, len(CAMERAS)))


def _speak_message(message):
    try:
        engine = pyttsx3.init(driverName='sapi5')
        engine.setProperty('volume', 1.0)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
    except Exception as speak_err:
        print(f"[VOICE] Speaking failed: {speak_err}")


def _voice_worker():
    while True:
        message = voice_queue.get()
        if message is None:
            voice_queue.task_done()
            break
        print(f"[VOICE] Speaking alert: {message}")
        _speak_message(message)
        voice_queue.task_done()


voice_thread = threading.Thread(target=_voice_worker, daemon=True)
voice_thread.start()

# Flask/Login Setup
app = Flask(__name__)
app.secret_key = "0c53fe07f67d4a64b702e3ea73dfc58b"
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
USERS = {"KSIT": generate_password_hash("ksit123")}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in USERS:
        return User(username)
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in USERS and check_password_hash(USERS[username], password):
            login_user(User(username))
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def voice_alert(message):
    try:
        voice_queue.put_nowait(message)
        print(f"[VOICE] Queued alert: {message}")
    except Full:
        print(f"[VOICE] Queue full, dropping alert: {message}")

def save_crime_video(cam_id, frames, status, location):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_location = location[:20].replace(' ', '_')
    fname = f"{SAVED_CLIPS_DIR}/{cam_id}_{status.replace(' ', '_')}_{safe_location}_{timestamp}.mp4"
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for f in frames:
        out.write(f)
    out.release()


def run_weapon_detection(frame, target_size):
    det_frame = cv2.resize(frame, target_size)
    det_h, det_w = det_frame.shape[:2]
    labels = set()
    boxes_norm = []
    box_labels = []
    confidences = []
    with weapon_lock:
        results = weapon_model.predict(
            source=det_frame,
            conf=DETECTION_CONF_THRESHOLD,
            iou=DETECTION_IOU_THRESHOLD,
            max_det=20,
            verbose=False
        )
    for box in results[0].boxes:
        classid = int(box.cls)
        if classid in CLASS_NAMES:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, coords)
            
            # filter by box area ratio to drop very small / very large spurious boxes
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = det_w * det_h
            area_ratio = box_area / frame_area if frame_area > 0 else 0
            if not (MIN_BOX_AREA_RATIO <= area_ratio <= MAX_BOX_AREA_RATIO):
                continue
            
            # Filter by aspect ratio (knives are usually taller than wide, not square/wide like fans)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_height > 0:
                aspect_ratio = box_width / box_height
                if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                    continue  # Skip boxes that are too wide (like fans) or too narrow
            
            # Filter out detections in top portion of frame (where fans/ceiling objects are)
            box_center_y = (y1 + y2) / 2
            if box_center_y < det_h * TOP_FRAME_EXCLUDE:
                continue  # Skip detections in top 15% of frame
            
            # Additional confidence check for knife class (class 1) - require higher confidence
            conf = float(box.conf.cpu().item())
            if classid == 1:  # Knife class
                if conf < 0.55:  # Require higher confidence for knives to reduce false positives
                    continue

            labels.add(CLASS_NAMES[classid])
            boxes_norm.append((x1 / det_w, y1 / det_h, x2 / det_w, y2 / det_h))
            box_labels.append(CLASS_NAMES[classid])
            confidences.append(conf)
    return {
        "weapon_detected": bool(labels),
        "weapon_labels": labels,
        "boxes_norm": boxes_norm,
        "box_labels": box_labels,
        "confidences": confidences,
        "det_size": target_size,
        "max_conf": max(confidences) if confidences else 0.0,
    }

def gen_frames(cam_id, cam_source):
    print(f"[INFO] Opening camera: {cam_source}")
    backend = cv2.CAP_DSHOW if isinstance(cam_source, int) and hasattr(cv2, "CAP_DSHOW") else None
    cap = cv2.VideoCapture(cam_source, backend) if backend else cv2.VideoCapture(cam_source)
    
    # Check if this is an IP camera
    is_ip_camera = isinstance(cam_source, str)
    
    # Set timeout and buffer settings for IP cameras (string sources)
    if is_ip_camera:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for IP cameras to reduce lag
    
    if not cap.isOpened():
        error_msg = f"[ERROR] Could not open camera {cam_id} at {cam_source}. "
        if is_ip_camera:
            error_msg += "Check: 1) Phone IP address is correct, 2) IP Webcam app is running, 3) Both devices on same WiFi network."
        print(error_msg)
        STATUSES[cam_id] = "Offline"
        return
    
    print(f"[INFO] Camera {cam_id} opened successfully (IP: {is_ip_camera})")
    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LOCAL_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LOCAL_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, LOCAL_CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, LOCAL_CAMERA_BUFFERSIZE)
    frame_count = 0
    fight_frame_buffer = []
    sequence_length = FIGHT_SEQUENCE_LENGTH
    fight_detected_prev = False
    cam_location = next((c["location"] for c in CAMERAS if c["id"] == cam_id), "Unknown Location")
    detection_task: Future | None = None
    detection_result = {
        "weapon_detected": False,
        "weapon_labels": set(),
        "boxes_norm": [],
        "box_labels": [],
        "confidences": [],
        "det_size": SMALL_FRAME_SIZE,
        "max_conf": 0.0,
    }
    detection_cycle = 0

    while True:
        loop_start = time.time()
        success, frame = cap.read()
        if not success:
            print(f"[ERROR] Camera {cam_id} stream ended or unavailable.")
            break

        LAST_FRAME_TIME[cam_id] = time.time()
        frame_count += 1

        # For IP cameras: skip frames to reduce processing lag
        # Initialize last_display_frame if not exists
        if 'last_display_frame' not in locals():
            last_display_frame = None
            
        if is_ip_camera and frame_count % IP_CAMERA_FRAME_SKIP != 0:
            # Skip processing this frame, but still yield the last frame to keep stream alive
            if last_display_frame is not None:
                _, buffer = cv2.imencode('.jpg', last_display_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        full_frame = frame.copy()
        VIDEO_BUFFERS[cam_id].append(full_frame)
        # For IP cameras: keep smaller buffer to reduce memory and lag
        max_buffer = VIDEO_CLIP_LENGTH // 2 if is_ip_camera else VIDEO_CLIP_LENGTH
        if len(VIDEO_BUFFERS[cam_id]) > max_buffer:
            VIDEO_BUFFERS[cam_id].pop(0)

        # For IP cameras: reduce fight detection frequency (every 3rd sequence for less lag)
        should_check_fight = not is_ip_camera or (frame_count // IP_CAMERA_FRAME_SKIP) % 3 == 0
        if should_check_fight:
            fight_frame = cv2.resize(full_frame, SMALL_FRAME_SIZE)
            fight_frame_buffer.append(fight_frame)
            if len(fight_frame_buffer) == sequence_length:
                fight_detected = fight_detect_on_frame(fight_model, fight_frame_buffer, sequence_length)
                fight_detected_prev = fight_detected
                fight_frame_buffer = []

        if detection_task and detection_task.done():
            detection_result = detection_task.result()
            detection_task = None

            weapon_detected = detection_result["weapon_detected"]
            weapon_labels = detection_result["weapon_labels"]
            weapon_text = " & ".join(sorted(weapon_labels)) if weapon_labels else ""
            max_conf = detection_result.get("max_conf", 0.0)
            WEAPON_CONF_HISTORY[cam_id].append(max_conf if weapon_detected else 0.0)
            smoothed_conf = sum(WEAPON_CONF_HISTORY[cam_id]) / len(WEAPON_CONF_HISTORY[cam_id]) if WEAPON_CONF_HISTORY[cam_id] else 0.0
            detection_result["smoothed_conf"] = smoothed_conf

            weapon_confirmed = weapon_detected and smoothed_conf >= MIN_ALERT_CONFIDENCE

            if weapon_confirmed:
                WEAPON_COUNT[cam_id] += 1
            else:
                WEAPON_COUNT[cam_id] = max(0, WEAPON_COUNT[cam_id] - 1)

            # Require 5 consecutive detections instead of 3 to reduce false positives (fans, etc.)
            if WEAPON_COUNT[cam_id] >= 5 and fight_detected_prev:
                STATUSES[cam_id] = f"CRIME HAPPENING! {weapon_text} & Fight detected!"
            elif WEAPON_COUNT[cam_id] >= 5:
                STATUSES[cam_id] = f"CRIME HAPPENING! {weapon_text} detected!"
            elif fight_detected_prev:
                STATUSES[cam_id] = "CRIME HAPPENING! crime detected!"
            else:
                STATUSES[cam_id] = "Monitoring"

            now = time.time()
            status_is_crime = STATUSES[cam_id] != "Monitoring"

            if status_is_crime and not PENDING_ALERTS[cam_id] and not ALERTS_SENT[cam_id]:
                if now - LAST_ALERT_TIME[cam_id] > COOLDOWN_SECONDS:
                    alert_message_voice = (
                        f"Crime detected! {STATUSES[cam_id]} at {cam_location}. "
                        f"Please check camera {cam_id} immediately."
                    )
                    voice_alert(alert_message_voice)
                    buffer_copy = list(VIDEO_BUFFERS[cam_id])
                    save_crime_video(cam_id, buffer_copy, STATUSES[cam_id], cam_location)
                    PENDING_ALERTS[cam_id] = True
                    LAST_ALERT_TIME[cam_id] = now
                    
                    # Track analytics
                    current_hour = datetime.datetime.now().hour
                    DETECTION_STATS["total_detections"] += 1
                    DETECTION_STATS["detections_by_hour"][current_hour] += 1
                    DETECTION_STATS["detections_by_camera"][cam_id] += 1
                    if weapon_text:
                        for weapon in weapon_labels:
                            DETECTION_STATS["weapon_detections"][weapon] += 1
                    if fight_detected_prev:
                        DETECTION_STATS["fight_detections"] += 1
                    
                    # Add to alert history
                    ALERT_HISTORY.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "camera": cam_id,
                        "status": STATUSES[cam_id],
                        "location": cam_location,
                        "weapons": list(weapon_labels) if weapon_labels else [],
                        "fight_detected": fight_detected_prev
                    })
                    # Keep only last 100 alerts
                    if len(ALERT_HISTORY) > 100:
                        ALERT_HISTORY.pop(0)
                else:
                    print(f"[INFO] Alert for {cam_id} in cooldown, no new alert.")
            elif not status_is_crime:
                ALERTS_SENT[cam_id] = False

        # Use different detection gap for IP cameras (less frequent due to network lag)
        detection_gap = DETECTION_FRAME_GAP_IP if is_ip_camera else DETECTION_FRAME_GAP
        detection_due = frame_count % detection_gap == 0
        suspicious = WEAPON_COUNT[cam_id] > 0 or fight_detected_prev
        if detection_due and detection_task is None:
            # For IP cameras: never use high-res, always use small detection size
            if is_ip_camera:
                target_size = IP_CAMERA_DETECTION_SIZE
            else:
                use_high_res = suspicious or (detection_cycle % HIGH_RES_DETECTION_GAP == 0)
                target_size = HIGH_RES_FRAME_SIZE if use_high_res else SMALL_FRAME_SIZE
            detection_task = DETECTION_EXECUTOR.submit(
                run_weapon_detection,
                full_frame.copy(),
                target_size
            )
            detection_cycle += 1

        # Use smaller stream size for IP cameras
        stream_size = IP_CAMERA_STREAM_SIZE if is_ip_camera else STREAM_FRAME_SIZE
        display_frame = cv2.resize(full_frame, stream_size)
        disp_h, disp_w = display_frame.shape[:2]
        for (nx1, ny1, nx2, ny2), label, conf in zip(
                detection_result.get("boxes_norm", []),
                detection_result.get("box_labels", []),
                detection_result.get("confidences", [])):
            x1 = int(nx1 * disp_w)
            y1 = int(ny1 * disp_h)
            x2 = int(nx2 * disp_w)
            y2 = int(ny2 * disp_h)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"{label}:{conf:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        cv2.putText(
            display_frame,
            STATUSES[cam_id],
            (10, disp_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        FRAME_TIMES[cam_id].append(time.time() - loop_start)
        avg_frame_time = sum(FRAME_TIMES[cam_id]) / len(FRAME_TIMES[cam_id]) if FRAME_TIMES[cam_id] else 0.0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        smoothed_conf_text = detection_result.get("smoothed_conf", detection_result.get("max_conf", 0.0))
        cv2.putText(
            display_frame,
            f"Conf: {smoothed_conf_text:.2f} | FPS: {fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (173, 216, 230),
            2,
            cv2.LINE_AA
        )

        # Store last frame for IP camera frame skipping
        last_display_frame = display_frame.copy()
        
        # For IP cameras: use much lower JPEG quality to reduce bandwidth and encoding time
        jpeg_quality = 65 if is_ip_camera else 90
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def make_video_feed(cam_id, cam_source):
    @login_required
    def video_feed():
        return Response(
            gen_frames(cam_id, cam_source),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    return video_feed

for cam in CAMERAS:
    app.add_url_rule(
        f"/video_feed_{cam['id']}",
        endpoint=f"video_feed_{cam['id']}",
        view_func=make_video_feed(cam["id"], cam["source"])
    )

@app.route('/')
@login_required
def index():
    return render_template('index.html', camera_ids=[cam["id"] for cam in CAMERAS], statuses=STATUSES)

@app.route('/status/<cam_id>')
@login_required
def get_status(cam_id):
    return STATUSES.get(cam_id, "Unknown")

@app.route('/pending_alerts')
@login_required
def get_pending_alerts():
    return jsonify(PENDING_ALERTS)

@app.route('/send_alert/<cam_id>', methods=['POST'])
@login_required
def send_alert(cam_id):
    cam_location = next((c["location"] for c in CAMERAS if c["id"] == cam_id), "Unknown Location")
    alert_message_sms = (
        f"{STATUSES[cam_id]} detected on camera {cam_id}!\n"
        f"Location: {cam_location}"
    )
    try:
        sendsms(SMS_TO, alert_message_sms, TWILIO_FROM)
        ALERTS_SENT[cam_id] = True
        PENDING_ALERTS[cam_id] = False
        return jsonify({'result': 'success'})
    except Exception as e:
        print(f"SMS send failed: {e}")
        return jsonify({'result': 'error', 'message': str(e)})

@app.route('/dismiss_alert/<cam_id>', methods=['POST'])
@login_required
def dismiss_alert(cam_id):
    PENDING_ALERTS[cam_id] = False
    ALERTS_SENT[cam_id] = False
    print(f"[INFO] Alert dismissed for {cam_id} by operator.")
    return jsonify({'result': 'dismissed'})

@app.route('/saved_clips_list')
@login_required
def saved_clips_list():
    files = os.listdir(SAVED_CLIPS_DIR)
    clips = []
    for f in files:
        if not f.endswith('.mp4'):
            continue
        path = os.path.join(SAVED_CLIPS_DIR, f)
        try:
            mtime = os.path.getmtime(path)
            dt = datetime.datetime.fromtimestamp(mtime)
            display_time = dt.strftime("%A, %d %b %Y • %I:%M %p")
        except Exception:
            display_time = "Unknown time"
        clips.append({
            "filename": f,
            "display_time": display_time,
            "timestamp": mtime if 'mtime' in locals() else None
        })
    clips.sort(key=lambda c: c.get("timestamp", 0), reverse=True)
    return jsonify({"clips": clips})

@app.route('/download_clip/<filename>')
@login_required
def download_clip(filename):
    return send_from_directory(SAVED_CLIPS_DIR, filename, as_attachment=True)

@app.route('/play_clip/<filename>')
@login_required
def play_clip(filename):
    """Stream video for browser playback"""
    return send_from_directory(SAVED_CLIPS_DIR, filename, as_attachment=False)

@app.route('/export_analytics_csv')
@login_required
def export_analytics_csv():
    """Export analytics data as CSV"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Total Detections', DETECTION_STATS["total_detections"]])
    writer.writerow(['Fight Detections', DETECTION_STATS["fight_detections"]])
    writer.writerow(['Alerts Sent', DETECTION_STATS["alerts_sent"]])
    writer.writerow(['System Uptime (hours)', round((time.time() - SYSTEM_START_TIME) / 3600, 2)])
    writer.writerow([])
    writer.writerow(['Weapon Type', 'Count'])
    for weapon, count in DETECTION_STATS["weapon_detections"].items():
        writer.writerow([weapon, count])
    writer.writerow([])
    writer.writerow(['Camera ID', 'Detections'])
    for cam_id, count in DETECTION_STATS["detections_by_camera"].items():
        writer.writerow([cam_id, count])
    writer.writerow([])
    writer.writerow(['Hour', 'Detections'])
    for hour, count in sorted(DETECTION_STATS["detections_by_hour"].items()):
        writer.writerow([f"{hour}:00", count])
    
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename=crimecatcher_analytics_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return response

@app.route('/export_alerts_csv')
@login_required
def export_alerts_csv():
    """Export alert history as CSV"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Timestamp', 'Camera', 'Status', 'Location', 'Weapons', 'Fight Detected'])
    for alert in ALERT_HISTORY:
        writer.writerow([
            alert.get('timestamp', ''),
            alert.get('camera', ''),
            alert.get('status', ''),
            alert.get('location', ''),
            ', '.join(alert.get('weapons', [])),
            'Yes' if alert.get('fight_detected', False) else 'No'
        ])
    
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename=crimecatcher_alerts_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return response

@app.route('/analytics')
@login_required
def get_analytics():
    """Get detection analytics and statistics"""
    uptime_seconds = time.time() - SYSTEM_START_TIME
    uptime_hours = uptime_seconds / 3600
    return jsonify({
        "total_detections": DETECTION_STATS["total_detections"],
        "weapon_detections": dict(DETECTION_STATS["weapon_detections"]),
        "fight_detections": DETECTION_STATS["fight_detections"],
        "alerts_sent": DETECTION_STATS["alerts_sent"],
        "detections_by_hour": dict(DETECTION_STATS["detections_by_hour"]),
        "detections_by_camera": dict(DETECTION_STATS["detections_by_camera"]),
        "uptime_hours": round(uptime_hours, 2),
        "system_start": datetime.datetime.fromtimestamp(SYSTEM_START_TIME).isoformat()
    })

@app.route('/alert_history')
@login_required
def get_alert_history():
    """Get alert history (last 50 alerts)"""
    return jsonify({"alerts": ALERT_HISTORY[-50:]})

@app.route('/system_health')
@login_required
def get_system_health():
    """Get system health metrics"""
    try:
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        else:
            cpu_percent = 0.0
            memory = type('obj', (object,), {'total': 0, 'used': 0, 'percent': 0.0})()
            disk = type('obj', (object,), {'total': 0, 'used': 0})()
        
        # Get GPU info if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "memory_reserved": round(torch.cuda.memory_reserved(0) / 1024**3, 2)
            }
        else:
            gpu_info = {"available": False}
        
        # Calculate average FPS for each camera
        camera_fps = {}
        for cam in CAMERAS:
            cam_id_str = cam["id"]
            if FRAME_TIMES[cam_id_str]:
                avg_time = sum(FRAME_TIMES[cam_id_str]) / len(FRAME_TIMES[cam_id_str])
                camera_fps[cam_id_str] = round(1.0 / avg_time if avg_time > 0 else 0, 1)
            else:
                camera_fps[cam_id_str] = 0.0
        
        disk_percent = round(disk.used / disk.total * 100, 1) if disk.total > 0 else 0.0
        
        return jsonify({
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / 1024**3, 2) if memory.total > 0 else 0,
                "used_gb": round(memory.used / 1024**3, 2) if memory.used > 0 else 0,
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / 1024**3, 2) if disk.total > 0 else 0,
                "used_gb": round(disk.used / 1024**3, 2) if disk.used > 0 else 0,
                "percent": disk_percent
            },
            "gpu": gpu_info,
            "device": DEVICE,
            "camera_fps": camera_fps,
            "online_cameras": sum(1 for s in STATUSES.values() if s != "Offline"),
            "total_cameras": len(CAMERAS)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/performance_metrics')
@login_required
def get_performance_metrics():
    """Get real-time performance metrics"""
    metrics = {}
    for cam in CAMERAS:
        cam_id = cam["id"]
        if FRAME_TIMES[cam_id]:
            avg_time = sum(FRAME_TIMES[cam_id]) / len(FRAME_TIMES[cam_id])
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            fps = 0.0
        
        metrics[cam_id] = {
            "fps": round(fps, 1),
            "status": STATUSES.get(cam_id, "Unknown"),
            "last_frame_age": round(time.time() - LAST_FRAME_TIME.get(cam_id, time.time()), 1),
            "weapon_count": WEAPON_COUNT.get(cam_id, 0),
            "avg_confidence": round(sum(WEAPON_CONF_HISTORY[cam_id]) / len(WEAPON_CONF_HISTORY[cam_id]), 2) if WEAPON_CONF_HISTORY[cam_id] else 0.0
        }
    return jsonify(metrics)

CAMERA_HEALTH_TIMEOUT = 10
def camera_health_monitor():
    while True:
        now = time.time()
        for cam in CAMERAS:
            cam_id = cam["id"]
            age = now - LAST_FRAME_TIME[cam_id]
            if age > CAMERA_HEALTH_TIMEOUT:
                if STATUSES[cam_id] != "Offline":
                    STATUSES[cam_id] = "Offline"
                    if not OFFLINE_ALERTED[cam_id]:
                        vmsg = f"Camera {cam_id} is offline. Please check immediately!"
                        print(f"[HEALTH ALERT] {vmsg}")
                        voice_alert(vmsg)
                        OFFLINE_ALERTED[cam_id] = True
            else:
                if STATUSES[cam_id] == "Offline":
                    STATUSES[cam_id] = "Monitoring"
                OFFLINE_ALERTED[cam_id] = False
        time.sleep(2)

t = threading.Thread(target=camera_health_monitor, daemon=True)
t.start()

if __name__ == '__main__':
    # Turn off Flask debug reloader to avoid multiple processes
    # trying to open the same camera (which can freeze or stop the stream),
    # and enable threaded mode so each camera/feed runs more smoothly.
    app.run(debug=False, threaded=True)
