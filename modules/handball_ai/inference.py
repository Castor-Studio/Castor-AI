# modules/handball_ai/inference.py

import time
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WINDOW_W, WINDOW_H = 960, 540

COCO_SPORTS_BALL_ID = 32
CONF_TH = 0.25
IMG_SZ = 640

DETECT_EVERY = 6
MAX_MISSES = 25
TRAIL_LEN = 0

MODEL_PATH = (Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt")


def _make_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

    raise RuntimeError("Trackers CSRT/KCF indisponibles (opencv-contrib-python requis).")


def _as_list4(b):
    """Convert bbox to a plain python list of length 4 if possible."""
    if b is None:
        return None
    if hasattr(b, "tolist"):  # numpy/torch
        b = b.tolist()

    # handle nested [[x1,y1,x2,y2]]
    if isinstance(b, (list, tuple)) and len(b) == 1 and isinstance(b[0], (list, tuple)) and len(b[0]) == 4:
        b = b[0]

    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return None
    return [b[0], b[1], b[2], b[3]]


def _sanitize_xyxy_to_xywh_int(bbox_xyxy, frame_w, frame_h):
    """
    Returns (x, y, w, h) as Python ints, clipped & validated.
    Returns None if invalid.
    """
    b = _as_list4(bbox_xyxy)
    if b is None:
        return None

    try:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        return None

    if any((math.isnan(v) or math.isinf(v)) for v in (x1, y1, x2, y2)):
        return None

    # clip
    x1 = max(0.0, min(x1, frame_w - 1.0))
    y1 = max(0.0, min(y1, frame_h - 1.0))
    x2 = max(0.0, min(x2, frame_w - 1.0))
    y2 = max(0.0, min(y2, frame_h - 1.0))

    w = x2 - x1
    h = y2 - y1
    if w <= 2.0 or h <= 2.0:
        return None

    # IMPORTANT: cast to *Python int* (pas numpy int)
    x = int(round(x1))
    y = int(round(y1))
    w = int(round(w))
    h = int(round(h))

    # ensure inside bounds after rounding
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    if x + w > frame_w:
        w = frame_w - x
    if y + h > frame_h:
        h = frame_h - y
    if w <= 2 or h <= 2:
        return None

    return (x, y, w, h)


def run(video_path: str, frameskip: int = 0, debug: bool = True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible : PyTorch CPU-only détecté.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    model = YOLO(str(MODEL_PATH))
    model.to("cuda")

    cv2.destroyAllWindows()
    cv2.startWindowThread()
    cv2.namedWindow("Handball", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Handball", WINDOW_W, WINDOW_H)

    tracker = None
    have_track = False
    misses = 0
    frame_idx = 0
    trail = []

    last_t = time.time()
    fps_smooth = 0.0

    def detect_ball(frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = model.predict(
            img, imgsz=IMG_SZ, conf=CONF_TH, device=0,
            half=True, verbose=False
        )[0]

        best_bbox = None
        best_conf = 0.0

        for box in res.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if cls == COCO_SPORTS_BALL_ID and conf > best_conf:
                xyxy = box.xyxy[0]
                if hasattr(xyxy, "tolist"):
                    xyxy = xyxy.tolist()
                best_bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                best_conf = conf

        return best_bbox, best_conf

    try:
        while True:
            frame = None
            for _ in range(frameskip + 1):
                ret, frame = cap.read()
                if not ret:
                    return
            if frame is None:
                return

            frame_idx += 1
            H, W = frame.shape[:2]

            need_detect = (not have_track) or (frame_idx % DETECT_EVERY == 0) or (misses >= MAX_MISSES)

            status = ""
            status_color = (255, 255, 255)

            if need_detect:
                bbox_xyxy, conf = detect_ball(frame)
                bb = _sanitize_xyxy_to_xywh_int(bbox_xyxy, W, H)

                if bb is None:
                    have_track = False
                    tracker = None
                    misses += 1
                    status = "DETECT none/invalid"
                    status_color = (0, 0, 255)
                else:
                    # DEBUG types (utile si ça casse encore)
                    if debug:
                        print("DEBUG tracker.init bb =", bb, "types:", [type(v) for v in bb])

                    tracker = _make_tracker()
                    try:
                        # bb must be a tuple of python ints
                        tracker.init(frame, tuple(bb))
                    except cv2.error as e:
                        # print extra debug to pinpoint exact types/source
                        print("ERROR on tracker.init")
                        print("  cv2 version:", cv2.__version__)
                        print("  frame dtype:", frame.dtype, "shape:", frame.shape, "contiguous:", frame.flags['C_CONTIGUOUS'])
                        print("  bb:", bb, "types:", [type(v) for v in bb])
                        print("  raw bbox_xyxy:", bbox_xyxy, "type:", type(bbox_xyxy))
                        raise

                    have_track = True
                    misses = 0

                    x, y, w, h = bb
                    cx, cy = x + w // 2, y + h // 2
                    trail.append((cx, cy))
                    trail = trail[-TRAIL_LEN:]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                    status = f"DETECT conf={conf:.2f}"
                    status_color = (0, 255, 0)

            else:
                ok, tbox = tracker.update(frame)
                if ok:
                    # tbox often floats; cast safely
                    x, y, w, h = tbox
                    x, y, w, h = int(round(float(x))), int(round(float(y))), int(round(float(w))), int(round(float(h)))

                    # validate & clip
                    x = max(0, min(x, W - 1))
                    y = max(0, min(y, H - 1))
                    w = max(1, min(w, W - x))
                    h = max(1, min(h, H - y))

                    cx, cy = x + w // 2, y + h // 2
                    trail.append((cx, cy))
                    trail = trail[-TRAIL_LEN:]

                    misses = 0
                    have_track = True

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

                    status = "TRACK"
                    status_color = (255, 255, 255)
                else:
                    have_track = False
                    tracker = None
                    misses += 1
                    status = "TRACK LOST"
                    status_color = (0, 0, 255)

            # Trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (255, 255, 255), 2)

            now = time.time()
            dt = now - last_t
            last_t = now
            fps = (1.0 / dt) if dt > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            if debug:
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f"FPS {fps_smooth:.1f} misses {misses} detectEvery {DETECT_EVERY}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            view = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            cv2.imshow("Handball", view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
