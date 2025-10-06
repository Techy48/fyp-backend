# main_api.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import uuid
import time
import json
import base64
import shutil
import tempfile
import traceback
import numpy as np
from typing import List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import joblib
from deepface import DeepFace
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier

# ---- PDF (ReportLab) ----
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)

# =========================
# App & Global Initializers
# =========================

app = FastAPI(title="AI-Powered Video Surveillance API", version="1.1.0")

# Allow React (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
REPORTS_DIR = os.path.abspath("./reports")
RESULT_FRAMES_DIR = os.path.abspath("./result_frames")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RESULT_FRAMES_DIR, exist_ok=True)

# Serve reports statically: GET /reports/<filename>.pdf
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# FaceNet embedder and KNN model (load once)
EMBEDDER: Optional[FaceNet] = None
KNN_MODEL: Optional[KNeighborsClassifier] = None

# Keep the SAME detector & thresholds as your working code
DETECTOR_BACKEND = "yunet"
CONF_THRESH = 0.85          # face confidence filter
KNN_DISTANCE_THRESHOLD = 0.70
UNKNOWN_EUCLIDEAN_THRESHOLD = 0.70   # when 1 suspect image
SAMPLE_PER_SECOND = True     # process exactly 1 frame per second

# ================
# Pydantic Schemas
# ================

class DetectionResult(BaseModel):
    timestamp: float                 # seconds (match your old API behavior)
    bbox: List[int]                  # [x, y, w, h]
    name: Optional[str] = None       # known mode label
    frame_image: Optional[str] = None  # data:image/jpeg;base64,...

class DetectionResponse(BaseModel):
    video_name: str
    mode: str                        # "known" or "unknown"
    report_link: Optional[str] = None
    detections: List[DetectionResult]


# ================
# Utility Functions
# ================

def init_models():
    global EMBEDDER, KNN_MODEL
    if EMBEDDER is None:
        EMBEDDER = FaceNet()
    if KNN_MODEL is None:
        try:
            KNN_MODEL = joblib.load("face_knn_model.pkl")
            print("Pre-trained KNN model loaded.")
        except Exception:
            print("Warning: face_knn_model.pkl not found or failed to load.")
            KNN_MODEL = None
    # Build DeepFace (keeps same detector behavior ready)
    DeepFace.build_model("VGG-Face")

def cv2_to_base64_bgr(img_bgr, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"

def draw_box_and_label(frame_bgr, bbox, label: Optional[str] = None):
    x, y, w, h = bbox
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x, y - th - 8), (x + tw + 6, y), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def save_frame_to_disk(frame_bgr, session_id: str) -> str:
    out_dir = os.path.join(RESULT_FRAMES_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"det_{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, frame_bgr)
    return fpath

def build_pdf_report(
    report_path: str,
    project_title: str,
    video_name: str,
    mode: str,
    detections: List[DetectionResult],
    local_frame_paths: List[str]
):
    doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{project_title}</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Mode: <b>{mode.capitalize()}</b>", styles["Normal"]))
    story.append(Paragraph(f"Video: <b>{video_name}</b>", styles["Normal"]))
    story.append(Paragraph(f"Date: <b>{time.strftime('%Y-%m-%d %H:%M:%S')}</b>", styles["Normal"]))
    story.append(Spacer(1, 12))

    data = [["#", "Timestamp (s)", "Bounding Box", "Name (Known Mode)"]]
    for i, det in enumerate(detections, start=1):
        bbox_str = f"[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]"
        name = det.name if det.name else "-"
        data.append([str(i), f"{det.timestamp:.2f}", bbox_str, name])

    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    if local_frame_paths:
        story.append(Paragraph("<b>Matched Frame Thumbnails</b>", styles["Heading3"]))
        story.append(Spacer(1, 8))
        row_imgs, row_count = [], 0
        for p in local_frame_paths:
            try:
                im = RLImage(p, width=2.7 * inch, height=1.7 * inch)
                row_imgs.append(im)
                row_count += 1
                if row_count == 2:
                    story.append(Table([[row_imgs[0], row_imgs[1]]], hAlign="LEFT"))
                    story.append(Spacer(1, 6))
                    row_imgs, row_count = [], 0
            except Exception:
                continue
        if row_imgs:
            story.append(Table([row_imgs], hAlign="LEFT"))
        story.append(Spacer(1, 12))

    story.append(Paragraph(f"Total Matches: <b>{len(detections)}</b>", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: Confidence/Distance metrics are intentionally omitted from this report.", styles["Italic"]))
    doc.build(story)

# ======================
# Core Processing Routines (MATCH OLD LOGIC)
# ======================

def process_known_mode(video_path: str, generate_report: bool, video_name: str) -> DetectionResponse:
    if KNN_MODEL is None:
        raise HTTPException(status_code=500, detail="Known suspect model is not loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = 0
    detections: List[DetectionResult] = []
    saved_frame_paths: List[str] = []
    session_id = uuid.uuid4().hex

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # === EXACT: process 1 frame per second ===
            if SAMPLE_PER_SECOND and frame_count % int(fps) != 0:
                continue

            try:
                faces = DeepFace.extract_faces(
                    frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                for face_obj in faces:
                    # Keep same confidence filter
                    if face_obj.get("confidence", 0) > CONF_THRESH:
                        # DeepFace 'face' is float [0..1]; convert EXACTLY as before
                        face_uint8 = (face_obj["face"] * 255).astype(np.uint8)
                        # Maintain same (BGR -> RGB) conversion you used
                        face_rgb = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2RGB)

                        emb = EMBEDDER.embeddings([face_rgb])[0]
                        # Nearest neighbor distance (n=1), SAME threshold
                        d = KNN_MODEL.kneighbors([emb])[0][0][0]
                        if d <= KNN_DISTANCE_THRESHOLD:
                            pred = KNN_MODEL.predict([emb])[0]
                            # Match old pretty-name formatting
                            label = str(pred).replace("_", " ").title()

                            # Timestamp in seconds, rounded (match old)
                            ts = round(frame_count / fps, 2)

                            # Draw + save preview
                            fa = face_obj.get("facial_area", {})
                            bbox = [int(fa.get("x", 0)), int(fa.get("y", 0)), int(fa.get("w", 0)), int(fa.get("h", 0))]
                            draw_box_and_label(frame, bbox, label)
                            b64 = cv2_to_base64_bgr(frame)
                            saved_path = save_frame_to_disk(frame, session_id)
                            saved_frame_paths.append(saved_path)

                            detections.append(DetectionResult(
                                timestamp=ts,
                                bbox=bbox,
                                name=label,
                                frame_image=b64
                            ))
            except Exception:
                continue
    finally:
        cap.release()

    report_link = None
    if generate_report:
        report_name = f"report_{os.path.splitext(os.path.basename(video_name))[0]}_{session_id}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_name)
        try:
            build_pdf_report(
                report_path=report_path,
                project_title="AI-Powered Video Surveillance - Detection Report",
                video_name=video_name,
                mode="known",
                detections=detections,
                local_frame_paths=saved_frame_paths
            )
            report_link = f"/reports/{report_name}"
        except Exception as e:
            print("PDF generation error:", e)
            traceback.print_exc()

    return DetectionResponse(
        video_name=video_name,
        mode="known",
        report_link=report_link,
        detections=detections
    )

def get_facenet_embedding_from_image_rgb(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Match your original helper: use FaceNet's .extract on the suspect image to detect a face,
    then compute embeddings on the cropped face.
    """
    try:
        results = EMBEDDER.extract(img_rgb, threshold=0.95)
        if not results:
            return None
        x, y, w, h = results[0]["box"]
        face = img_rgb[y:y + h, x:x + w]
        return EMBEDDER.embeddings([face])[0]
    except Exception:
        return None

def process_unknown_mode(
    video_path: str,
    suspect_images: List[UploadFile],
    generate_report: bool,
    video_name: str
) -> DetectionResponse:

    # Build suspect embeddings EXACTLY like your old pipeline
    suspect_embeddings: List[np.ndarray] = []
    for uf in suspect_images:
        data = uf.file.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        emb = get_facenet_embedding_from_image_rgb(img_rgb)
        if emb is not None:
            suspect_embeddings.append(emb)

    if not suspect_embeddings:
        raise HTTPException(status_code=400, detail="No clear face found in suspect images.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = 0
    detections: List[DetectionResult] = []
    saved_frame_paths: List[str] = []
    session_id = uuid.uuid4().hex

    # === EXACT: branching logic ===
    use_single = (len(suspect_embeddings) == 1)
    temp_knn: Optional[KNeighborsClassifier] = None
    if not use_single:
        temp_knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        temp_knn.fit(suspect_embeddings, ["Suspect"] * len(suspect_embeddings))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 1 FPS sampling
            if SAMPLE_PER_SECOND and frame_count % int(fps) != 0:
                continue

            try:
                faces = DeepFace.extract_faces(
                    frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                for face_obj in faces:
                    if face_obj.get("confidence", 0) > CONF_THRESH:
                        face_uint8 = (face_obj["face"] * 255).astype(np.uint8)
                        face_rgb = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2RGB)
                        emb = EMBEDDER.embeddings([face_rgb])[0]

                        match = False
                        label_to_draw = None

                        if use_single:
                            # Euclidean distance to the one suspect emb (match your old code)
                            d = float(np.linalg.norm(suspect_embeddings[0] - emb))
                            if d <= UNKNOWN_EUCLIDEAN_THRESHOLD:
                                match = True
                                label_to_draw = "suspect"
                        else:
                            # Temporary 1-NN with only suspect images (no negatives), SAME threshold logic
                            d = temp_knn.kneighbors([emb])[0][0][0]
                            if d <= KNN_DISTANCE_THRESHOLD:
                                match = True
                                label_to_draw = "suspect"

                        if match:
                            ts = round(frame_count / fps, 2)
                            fa = face_obj.get("facial_area", {})
                            bbox = [int(fa.get("x", 0)), int(fa.get("y", 0)), int(fa.get("w", 0)), int(fa.get("h", 0))]
                            draw_box_and_label(frame, bbox, label_to_draw)
                            b64 = cv2_to_base64_bgr(frame)
                            saved_path = save_frame_to_disk(frame, session_id)
                            saved_frame_paths.append(saved_path)

                            detections.append(DetectionResult(
                                timestamp=ts,
                                bbox=bbox,
                                name=None,
                                frame_image=b64
                            ))
            except Exception:
                continue
    finally:
        cap.release()

    report_link = None
    if generate_report:
        report_name = f"report_{os.path.splitext(os.path.basename(video_name))[0]}_{session_id}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_name)
        try:
            build_pdf_report(
                report_path=report_path,
                project_title="AI-Powered Video Surveillance - Detection Report",
                video_name=video_name,
                mode="unknown",
                detections=detections,
                local_frame_paths=saved_frame_paths
            )
            report_link = f"/reports/{report_name}"
        except Exception as e:
            print("PDF generation error:", e)
            traceback.print_exc()

    return DetectionResponse(
        video_name=video_name,
        mode="unknown",
        report_link=report_link,
        detections=detections
    )

# ============
# API Endpoints
# ============

@app.get("/")
def root():
    return {"message": "API is running."}

@app.get("/health")
def healthcheck():
    return {"status": "ok"}

@app.post("/detect_known_suspect", response_model=DetectionResponse)
def detect_known_suspect(
    video: UploadFile = File(...),
    generate_report: bool = Form(False)
):
    init_models()
    if KNN_MODEL is None:
        raise HTTPException(status_code=500, detail="Known suspect model is not loaded.")

    # Save uploaded video to a temp file
    tmpdir = tempfile.mkdtemp(prefix="vidproc_")
    video_path = os.path.join(tmpdir, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        response = process_known_mode(
            video_path=video_path,
            generate_report=generate_report,
            video_name=video.filename
        )
        return JSONResponse(status_code=200, content=json.loads(response.json()))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        try:
            os.remove(video_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

@app.post("/detect_unknown_suspect", response_model=DetectionResponse)
def detect_unknown_suspect(
    video: UploadFile = File(...),
    suspect_images: List[UploadFile] = File(...),
    generate_report: bool = Form(False)
):
    init_models()

    tmpdir = tempfile.mkdtemp(prefix="vidproc_")
    video_path = os.path.join(tmpdir, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        response = process_unknown_mode(
            video_path=video_path,
            suspect_images=suspect_images,
            generate_report=generate_report,
            video_name=video.filename
        )
        return JSONResponse(status_code=200, content=json.loads(response.json()))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        try:
            os.remove(video_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
