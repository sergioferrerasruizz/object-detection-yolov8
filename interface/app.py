import os
import threading
import time
from collections import Counter

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


INTERFACE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INTERFACE_DIR, os.pardir))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo_custom", "weights", "best.pt")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


class YoloVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        max_det: int,
    ):
        self.model = load_model(model_path)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        self._lock = threading.Lock()
        self._last_counts: dict[str, int] = {}

    @property
    def last_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_counts)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img_bgr,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )

        r0 = results[0]
        annotated_bgr = r0.plot()

        counts: Counter[str] = Counter()
        if r0.boxes is not None and getattr(r0.boxes, "cls", None) is not None:
            cls_ids = r0.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            for cid in cls_ids:
                name = self.model.names.get(cid, str(cid))
                counts[name] += 1

        with self._lock:
            self._last_counts = dict(counts)

        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


def render_counts(counts: dict[str, int]) -> None:
    if not counts:
        st.write("Sin detecciones")
        return

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    st.write("Objetos detectados:")
    for name, n in items:
        st.write(f"- {name}: {n}")


def infer_image(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float, max_det: int):
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    r0 = results[0]
    annotated_bgr = r0.plot()

    counts: Counter[str] = Counter()
    if r0.boxes is not None and getattr(r0.boxes, "cls", None) is not None:
        cls_ids = r0.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        for cid in cls_ids:
            name = model.names.get(cid, str(cid))
            counts[name] += 1

    return annotated_bgr, dict(counts)


st.set_page_config(page_title="YOLOv8 - Detección en tiempo real", layout="wide")

st.title("YOLOv8 - Detección de objetos")

with st.sidebar:
    st.header("Configuración")

    model_path = st.text_input("Ruta del modelo (.pt)", value=DEFAULT_MODEL_PATH)
    conf = st.slider("Confianza (conf)", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("IoU (nms)", min_value=0.05, max_value=0.95, value=0.45, step=0.05)
    max_det = st.slider("Máx. detecciones", min_value=1, max_value=300, value=100, step=1)

    mode = st.radio("Modo", options=["Webcam (tiempo real)", "Imagen", "Vídeo"], index=0)

if not os.path.exists(model_path):
    st.error(f"No existe el modelo en: {model_path}")
    st.stop()

model = load_model(model_path)

if mode == "Webcam (tiempo real)":
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        ctx = webrtc_streamer(
            key="yolo-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=lambda: YoloVideoProcessor(
                model_path=model_path,
                conf=conf,
                iou=iou,
                max_det=max_det,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_right:
        st.subheader("Detecciones")
        placeholder = st.empty()

        if ctx.video_processor:
            placeholder.write("Iniciando...")
            while ctx.state.playing:
                vp = ctx.video_processor
                if vp is None:
                    placeholder.write("Esperando a la webcam...")
                    time.sleep(0.05)
                    continue

                counts = vp.last_counts
                with placeholder.container():
                    render_counts(counts)
                time.sleep(0.05)
        else:
            st.write("Pulsa 'START' para activar la webcam")

elif mode == "Imagen":
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.stop()

    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("No se pudo leer la imagen")
        st.stop()

    annotated_bgr, counts = infer_image(model, img_bgr, conf=conf, iou=iou, max_det=max_det)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.image(annotated_rgb, caption="Resultado", use_container_width=True)
    with col_right:
        render_counts(counts)

else:  # Vídeo
    uploaded = st.file_uploader("Sube un vídeo", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.stop()

    tmp_path = os.path.join(INTERFACE_DIR, "_tmp_video_input")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("No se pudo abrir el vídeo")
        st.stop()

    stframe = st.empty()
    det_placeholder = st.empty()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        annotated_bgr, counts = infer_image(model, frame_bgr, conf=conf, iou=iou, max_det=max_det)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
        with det_placeholder.container():
            render_counts(counts)

    cap.release()
    try:
        os.remove(tmp_path)
    except OSError:
        pass
