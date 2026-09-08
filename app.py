from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE = BASE_DIR / "factory_ai_manual.db"
MODEL_DIR = BASE_DIR / "models"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


def connect():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_database():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL, file_name TEXT NOT NULL, file_path TEXT NOT NULL,
          duration REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, title TEXT NOT NULL,
          start_time REAL NOT NULL, end_time REAL NOT NULL,
          description_ja TEXT NOT NULL DEFAULT '', feature_json TEXT NOT NULL DEFAULT '[]',
          UNIQUE(video_id, position)
        );
        CREATE TABLE IF NOT EXISTS translations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          process_id INTEGER NOT NULL REFERENCES processes(id) ON DELETE CASCADE,
          language_code TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', description TEXT NOT NULL,
          UNIQUE(process_id, language_code)
        );
        CREATE TABLE IF NOT EXISTS training_examples (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          label TEXT NOT NULL, feature_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(processes)")}
        if "feature_json" not in columns:
            db.execute("ALTER TABLE processes ADD COLUMN feature_json TEXT NOT NULL DEFAULT '[]'")
        translation_columns = {row[1] for row in db.execute("PRAGMA table_info(translations)")}
        if "title" not in translation_columns:
            db.execute("ALTER TABLE translations ADD COLUMN title TEXT NOT NULL DEFAULT ''")


init_database()


def ensure_model(path: Path, url: str) -> str:
    if not path.exists():
        try:
            urlretrieve(url, path)
        except OSError as error:
            path.unlink(missing_ok=True)
            raise ValueError("MediaPipeモデルを取得できません。初回のみネット接続が必要です。") from error
    return str(path)


def detect_process_ranges(path: Path) -> tuple[float, list[dict]]:
    """Find a variable number of boundaries from image, arm and hand motion."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("動画を読み込めませんでした。MP4をお試しください。")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames else 0
    if duration < 3:
        capture.release()
        raise ValueError("3秒以上の動画を使用してください。")
    pose_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=ensure_model(POSE_MODEL, POSE_URL)), running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=.25, min_tracking_confidence=.25)
    hand_options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=ensure_model(HAND_MODEL, HAND_URL)), running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=.25, min_tracking_confidence=.25)
    interval = 0.5
    samples, previous_gray, previous_body = [], None, None
    with vision.PoseLandmarker.create_from_options(pose_options) as pose, vision.HandLandmarker.create_from_options(hand_options) as hands:
        for time in np.arange(0, duration, interval):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time * 1000))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 27))
            image_motion = 0.0 if previous_gray is None else float(np.mean(cv2.absdiff(gray, previous_gray)) / 255)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time * 1000)
            pose_result, hand_result = pose.detect_for_video(image, timestamp), hands.detect_for_video(image, timestamp)
            body = []
            if pose_result.pose_landmarks:
                for index in (11, 12, 13, 14, 15, 16):
                    point = pose_result.pose_landmarks[0][index]
                    body.extend((point.x, point.y))
            body.extend(value for hand in hand_result.hand_landmarks for point in hand for value in (point.x, point.y))
            body_array = np.asarray(body[:84], dtype=float)
            body_motion = 0.0
            if previous_body is not None and len(body_array) == len(previous_body) and len(body_array):
                body_motion = float(np.mean(np.abs(body_array - previous_body)))
            hand_count = len(hand_result.hand_landmarks)
            samples.append({"time": float(time), "image": image_motion, "body": body_motion, "hands": hand_count})
            previous_gray, previous_body = gray, body_array
    capture.release()
    if len(samples) < 5:
        return duration, [{"start": 0.0, "end": duration, "feature": [0, 0, 0]}]
    raw = np.asarray([s["image"] + s["body"] * 5 for s in samples])
    smooth = np.convolve(raw, np.ones(3) / 3, mode="same")
    change = np.abs(np.diff(smooth, prepend=smooth[0]))
    threshold = max(float(np.percentile(change, 75)), .008)
    minimum_gap, boundaries, last = 2.0, [0.0], 0.0
    for index in np.argsort(change)[::-1]:
        time = samples[int(index)]["time"]
        if change[index] < threshold or time < minimum_gap or duration - time < minimum_gap:
            continue
        if all(abs(time - boundary) >= minimum_gap for boundary in boundaries):
            boundaries.append(round(time, 2))
            last = time
        if len(boundaries) >= 12:
            break
    boundaries.append(round(duration, 2))
    boundaries.sort()
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        selected = [s for s in samples if start <= s["time"] < end]
        if end - start < 1.5 or not selected:
            continue
        feature = [float(np.mean([s["image"] for s in selected])), float(np.mean([s["body"] for s in selected])), float(np.mean([s["hands"] for s in selected]))]
        segments.append({"start": start, "end": end, "feature": feature})
    return duration, segments or [{"start": 0.0, "end": duration, "feature": [0, 0, 0]}]


def predict_labels(segments: list[dict]) -> list[str]:
    allowed = ["ねじ締め", "ねじの締付確認", "不適合確認"]
    with connect() as db:
        examples = db.execute("SELECT label, feature_json FROM training_examples").fetchall()
    learned = [(row["label"], np.asarray(json.loads(row["feature_json"]))) for row in examples if row["label"] in allowed]
    labels = []
    activity = [float(item["feature"][0]) + float(item["feature"][1]) * 5 for item in segments]
    median_activity = float(np.median(activity)) if activity else 0
    for segment in segments:
        feature = np.asarray(segment["feature"])
        if len(learned) >= 6:
            nearest = sorted(((float(np.linalg.norm(feature - saved)), label) for label, saved in learned))[:3]
            labels.append(max(allowed, key=lambda label: sum(1 / (distance + .001) for distance, found in nearest if found == label)))
        else:
            movement = float(feature[0]) + float(feature[1]) * 5
            hands = float(feature[2])
            if hands >= .5 and movement >= median_activity:
                labels.append("ねじ締め")
            elif hands >= .5 or movement >= median_activity * .65:
                labels.append("ねじの締付確認")
            else:
                labels.append("不適合確認")
    return labels


def generate_processes_with_phi4(manual_title: str, segments: list[dict]) -> list[dict]:
    labels = predict_labels(segments)
    timing = [{"number": i + 1, "start": item["start"], "end": item["end"], "seconds": round(item["end"] - item["start"], 1), "label": labels[i]} for i, item in enumerate(segments)]
    prompt = f"""あなたは製造現場の標準作業書を作る専門家です。
マニュアル名: {manual_title}
OpenCVが映像変化から検出した工程候補: {json.dumps(timing, ensure_ascii=False)}
工程名は候補のlabelをそのまま使ってください。labelは「ねじ締め」「ねじの締付確認」「不適合確認」「判定保留」のいずれかです。
工程数を変えないでください。各工程の説明を日本語で1文、30文字以内で作ってください。推測した固有部品名や数値は書かないでください。
JSONのみを返してください。形式: {{"processes":[{{"number":1,"title":"...","description":"..."}}]}}"""
    try:
        data = json.dumps({"model": "phi4", "prompt": prompt, "stream": False, "format": "json", "keep_alive": "10m", "options": {"temperature": 0.1, "num_predict": 180}}).encode("utf-8")
        req = Request("http://127.0.0.1:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=180) as response:
            result = json.loads(json.loads(response.read())["response"])["processes"]
    except Exception as error:
        raise ValueError("Ollama + Phi-4で工程文を生成できませんでした。Ollamaを起動し、ollama run phi4 を実行してください。") from error
    if not isinstance(result, list) or len(result) != len(segments):
        raise ValueError("Phi-4の生成結果を工程データとして読み取れませんでした。もう一度お試しください。")
    for index, item in enumerate(result):
        item["title"] = labels[index]
        description = str(item.get("description", "")).strip().replace("\n", " ")
        if not description:
            raise ValueError("Phi-4が説明文を生成できませんでした。もう一度お試しください。")
        item["description"] = description[:30]
    return result


@app.get("/")
def home():
    with connect() as db:
        videos = db.execute("SELECT v.*, COUNT(p.id) process_count FROM videos v LEFT JOIN processes p ON p.video_id=v.id GROUP BY v.id ORDER BY v.id DESC").fetchall()
    return render_template("index.html", videos=videos)


@app.post("/api/videos")
def create_video():
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify(error="動画を選択してください。"), 400
    extension = Path(uploaded.filename).suffix.lower()
    if extension not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        return jsonify(error="MP4 / MOV / WebM / MKV / AVIに対応しています。"), 400
    safe_name = secure_filename(uploaded.filename) or f"video{extension}"
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    uploaded.save(stored)
    title = request.form.get("title", "").strip() or Path(uploaded.filename).stem
    try:
        duration, ranges = detect_process_ranges(stored)
    except ValueError:
        capture = cv2.VideoCapture(str(stored))
        fps, frames = capture.get(cv2.CAP_PROP_FPS) or 30, capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frames / fps if frames else 0
        capture.release()
        if duration <= 0:
            stored.unlink(missing_ok=True)
            return jsonify(error="動画を読み込めませんでした。別のMP4動画をお試しください。"), 400
        ranges = [{"start": 0.0, "end": round(duration, 2), "feature": [0, 0, 0]}]
    try:
        generated = generate_processes_with_phi4(title, ranges)
    except ValueError:
        generated = [{"title": "判定保留", "description": "映像を確認して作業内容を登録してください。"} for _ in ranges]
    with connect() as db:
        cursor = db.execute("INSERT INTO videos(title,file_name,file_path,duration,created_at) VALUES(?,?,?,?,?)", (title, uploaded.filename, str(stored), duration, datetime.now(timezone.utc).isoformat()))
        video_id = cursor.lastrowid
        for position, (segment, text) in enumerate(zip(ranges, generated), 1):
            db.execute("INSERT INTO processes(video_id,position,title,start_time,end_time,description_ja,feature_json) VALUES(?,?,?,?,?,?,?)", (video_id, position, str(text.get("title", "判定保留")).strip(), segment["start"], segment["end"], str(text.get("description", "")).strip(), json.dumps(segment["feature"])))
    return jsonify(id=video_id, manual_url=f"/manuals/{video_id}")


def load_manual(video_id):
    with connect() as db:
        video = db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        processes = db.execute("SELECT * FROM processes WHERE video_id=? ORDER BY position", (video_id,)).fetchall()
    return video, processes


@app.get("/manuals/<int:video_id>/edit")
def edit_manual(video_id):
    video, processes = load_manual(video_id)
    if not video:
        return "マニュアルが見つかりません。", 404
    return render_template("editor.html", video=video, processes=processes)


@app.get("/manuals/<int:video_id>")
def view_manual(video_id):
    language_code = request.args.get("lang", "ja")
    if language_code not in {"ja", "en", "vi", "zh"}:
        language_code = "ja"
    with connect() as db:
        video = db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if language_code == "ja":
            processes = db.execute("SELECT * FROM processes WHERE video_id=? ORDER BY position", (video_id,)).fetchall()
        else:
            processes = db.execute("""
                SELECT p.id, p.video_id, p.position, p.start_time, p.end_time,
                       p.feature_json,
                       COALESCE(NULLIF(t.title, ''), p.title) AS title,
                       COALESCE(NULLIF(t.description, ''), p.description_ja) AS description_ja
                FROM processes AS p
                LEFT JOIN translations AS t
                  ON t.process_id = p.id AND t.language_code = ?
                WHERE p.video_id = ?
                ORDER BY p.position
            """, (language_code, video_id)).fetchall()
    if not video:
        return "マニュアルが見つかりません。", 404
    return render_template("manual.html", video=video, processes=processes, language_code=language_code)


def ensure_argos_pair(source_code: str, target_code: str) -> None:
    from argostranslate import package, translate

    try:
        if translate.get_translation_from_codes(source_code, target_code) is not None:
            return
    except Exception:
        pass
    package.update_package_index()
    available = package.get_available_packages()
    direct = next((item for item in available if item.from_code == source_code and item.to_code == target_code), None)
    if direct:
        package.install_from_path(direct.download())
        translate.get_installed_languages.cache_clear()
        return
    # Argos can pivot automatically. Install Japanese→English and
    # English→target when a direct package is unavailable.
    pairs = [(source_code, "en"), ("en", target_code)]
    for from_code, to_code in pairs:
        try:
            if translate.get_translation_from_codes(from_code, to_code) is not None:
                continue
        except Exception:
            pass
        model = next((item for item in available if item.from_code == from_code and item.to_code == to_code), None)
        if model is None:
            raise ValueError(f"Argos翻訳モデル {from_code}→{to_code} が見つかりません。")
        package.install_from_path(model.download())
        translate.get_installed_languages.cache_clear()


def argos_translate_text(text: str, target_code: str) -> str:
    from argostranslate import translate

    ensure_argos_pair("ja", target_code)
    translate.get_installed_languages.cache_clear()
    try:
        translator = translate.get_translation_from_codes("ja", target_code)
    except (AttributeError, StopIteration):
        translator = None
    if translator is not None:
        return translator.translate(text)
    # Explicit pivot is a fallback for Argos versions that do not rebuild the
    # composite Japanese→English→target route immediately after installation.
    first = translate.get_translation_from_codes("ja", "en")
    second = translate.get_translation_from_codes("en", target_code)
    if first is None or second is None:
        raise ValueError("インストール済み翻訳モデルを読み込めません。")
    return second.translate(first.translate(text))


@app.post("/manuals/<int:video_id>/translate")
def translate_manual(video_id):
    language_code = request.form.get("language", "")
    languages = {"en": "英語", "vi": "ベトナム語", "zh": "中国語"}
    if language_code not in languages:
        return redirect(url_for("view_manual", video_id=video_id))
    video, processes = load_manual(video_id)
    if not video or not processes:
        return "翻訳する工程がありません。", 400
    try:
        ensure_argos_pair("ja", language_code)
        translated = [{"id": row["id"], "title": argos_translate_text(row["title"], language_code), "description": argos_translate_text(row["description_ja"], language_code)} for row in processes]
    except Exception as error:
        return f"Argos Translateで翻訳できませんでした: {error}", 503
    with connect() as db:
        for item in translated:
            db.execute("INSERT INTO translations(process_id,language_code,title,description) VALUES(?,?,?,?) ON CONFLICT(process_id,language_code) DO UPDATE SET title=excluded.title,description=excluded.description", (int(item["id"]), language_code, str(item["title"]).strip(), str(item["description"]).strip()))
    return redirect(url_for("view_manual", video_id=video_id, lang=language_code))


@app.get("/api/videos/<int:video_id>/stream")
def stream_video(video_id):
    with connect() as db:
        video = db.execute("SELECT file_path FROM videos WHERE id=?", (video_id,)).fetchone()
    if not video or not Path(video["file_path"]).exists():
        return jsonify(error="動画が見つかりません。"), 404
    return send_file(video["file_path"], conditional=True)


@app.post("/api/videos/<int:video_id>/processes")
def create_process(video_id):
    payload = request.get_json(silent=True) or {}
    try:
        title = str(payload["title"]).strip()
        start, end = round(float(payload["start_time"]), 2), round(float(payload["end_time"]), 2)
        if not title or start < 0 or end <= start:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return jsonify(error="工程名と正しい開始・終了位置を入力してください。"), 400
    with connect() as db:
        video = db.execute("SELECT duration FROM videos WHERE id=?", (video_id,)).fetchone()
        if not video:
            return jsonify(error="動画が見つかりません。"), 404
        if video["duration"] and end > video["duration"] + .5:
            return jsonify(error="終了位置が動画の長さを超えています。"), 400
        position = db.execute("SELECT COALESCE(MAX(position),0)+1 FROM processes WHERE video_id=?", (video_id,)).fetchone()[0]
        cursor = db.execute("INSERT INTO processes(video_id,position,title,start_time,end_time,description_ja) VALUES(?,?,?,?,?,?)", (video_id, position, title, start, end, f"{title}を行います。作業後に状態を確認してください。"))
    return jsonify(id=cursor.lastrowid, position=position)


@app.patch("/api/processes/<int:process_id>")
def update_process(process_id):
    payload = request.get_json(silent=True) or {}
    title, description = str(payload.get("title", "")).strip(), str(payload.get("description_ja", "")).strip()
    if not title:
        return jsonify(error="工程名を入力してください。"), 400
    with connect() as db:
        current = db.execute("SELECT title,feature_json FROM processes WHERE id=?", (process_id,)).fetchone()
        if not current:
            return jsonify(error="工程が見つかりません。"), 404
        cursor = db.execute("UPDATE processes SET title=?,description_ja=? WHERE id=?", (title, description, process_id))
        if title in {"ねじ締め", "ねじの締付確認", "不適合確認"} and current["feature_json"] != "[]":
            db.execute("INSERT INTO training_examples(label,feature_json,created_at) VALUES(?,?,?)", (title, current["feature_json"], datetime.now(timezone.utc).isoformat()))
    return jsonify(ok=bool(cursor.rowcount))


@app.delete("/api/processes/<int:process_id>")
def delete_process(process_id):
    with connect() as db:
        row = db.execute("SELECT video_id FROM processes WHERE id=?", (process_id,)).fetchone()
        if not row:
            return jsonify(error="工程が見つかりません。"), 404
        db.execute("DELETE FROM processes WHERE id=?", (process_id,))
        rows = db.execute("SELECT id FROM processes WHERE video_id=? ORDER BY position", (row["video_id"],)).fetchall()
        for position, item in enumerate(rows, 1):
            db.execute("UPDATE processes SET position=? WHERE id=?", (position, item["id"]))
    return jsonify(ok=True)


@app.post("/api/processes/<int:process_id>/ai")
def ai_description(process_id):
    with connect() as db:
        process = db.execute("SELECT title FROM processes WHERE id=?", (process_id,)).fetchone()
    if not process:
        return jsonify(error="工程が見つかりません。"), 404
    prompt = f"製造現場の動画マニュアルです。工程『{process['title']}』の説明を日本語で1文、30文字以内で書いてください。"
    try:
        data = json.dumps({"model": "phi4", "prompt": prompt, "stream": False}).encode()
        req = Request("http://127.0.0.1:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=90) as response:
            text = json.loads(response.read())["response"].strip()[:30]
    except Exception:
        return jsonify(error="Ollamaに接続できません。Ollamaを起動して phi4 を準備してください。"), 503
    with connect() as db:
        db.execute("UPDATE processes SET description_ja=? WHERE id=?", (text, process_id))
    return jsonify(description_ja=text)


if __name__ == "__main__":
    app.run(debug=True, port=5051)
