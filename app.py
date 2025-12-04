import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
from neural_net import NN, one_hot_encode
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="NumPy MNIST Classifier",
    page_icon="🧠",
    layout="wide",
)

APP_CSS = """
<style>
body {
    background: linear-gradient(180deg, #0f0f0f 0%, #072d2d 60%, #051b11 100%);
    color: #f1f5f9;
    min-height: 100vh;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #4c1d95, #1d4ed8);
    color: #f8fafc;
    padding: 1.5rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.25rem;
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.35);
}
.hero h1 {
    margin: 0;
}
.card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(15, 23, 42, 0.08);
    padding: 1.25rem;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.2);
}
.pill {
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    font-size: 0.9rem;
    gap: 0.35rem;
}
.metric {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.metric-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.85rem;
    color: #94a3b8;
}
.confidence-display {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1rem 1.25rem;
    border-radius: 14px;
    background: linear-gradient(120deg, #0f172a, #1e3a8a);
    box-shadow: inset 0 0 25px rgba(15, 23, 42, 0.25);
    margin-bottom: 0.75rem;
}
.confidence-digit {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    color: #f8fafc;
}
.confidence-info span {
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(248, 250, 252, 0.7);
}
.confidence-info strong {
    display: block;
    font-size: 1.35rem;
    color: #fff;
    margin-top: 0.2rem;
}
.top-guess-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.top-guess {
    background: rgba(15, 23, 42, 0.02);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 14px;
    padding: 0.8rem;
    text-align: center;
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
}
.top-guess.accent {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: #fff;
    border: none;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.5);
}
.top-guess-rank {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}
.top-guess.accent .top-guess-rank {
    color: rgba(255, 255, 255, 0.85);
}
.top-guess-digit {
    font-size: 2rem;
    font-weight: 600;
    margin: 0.3rem 0;
}
.top-guess-confidence {
    font-size: 1rem;
    font-weight: 600;
}
.chart-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
    color: #0f172a;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = Path(__file__).with_name("mnist_weights.npz")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    nn = NN(
        X=np.zeros((784, 1)),
        y=np.zeros((10, 1)),
        X_test=np.zeros((784, 1)),
        y_test=np.zeros((10, 1)),
        activation="relu",
        num_labels=10,
        architecture=[128, 32],
    )
    try:
        nn.load(str(MODEL_PATH))
    except FileNotFoundError:
        st.error(f"Could not find model weights at {MODEL_PATH.resolve()}")
        st.stop()
    except Exception as exc:  # pragma: no cover - displayed inside Streamlit
        st.error(f"Failed to load model weights: {exc}")
        st.stop()
    return nn

nn = load_model()

# -----------------------------
# STREAMLIT APP UI
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="pill">⚡ NumPy Neural Network · MNIST</div>
        <h1 style="margin-top:0.5rem;">Sketch a digit, let the AI read it.</h1>
        <p>Use the canvas to draw any digit (0-9). The NumPy network predicts the number in real time and shows the confidence distribution.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Creative Controls ✏️")
st.sidebar.markdown("Tune the brush and smoothing to mimic the MNIST writing style.")
stroke_w = st.sidebar.slider("Stroke Width", 5, 40, 20)
blur_amount = st.sidebar.slider("Smoothing (Gaussian Blur)", 0, 3, 1)

st.sidebar.divider()
st.sidebar.subheader("App Tips")
st.sidebar.markdown(
    "- Keep strokes centered for best accuracy.\n"
    "- Use the smoothing slider to soften jagged edges.\n"
    "- Click **Clear Canvas** whenever you want a fresh start."
)

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

header_col1, header_col2 = st.columns([2, 1])
with header_col1:
    st.markdown("### Step 1 · Sketch Your Digit")
with header_col2:
    if st.button("🧼 Clear Canvas", use_container_width=True):
        st.session_state["canvas_key"] += 1
        st.rerun()

workspace_col1, workspace_col2 = st.columns([2.2, 1])

with workspace_col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="white",
        stroke_color="black",
        stroke_width=stroke_w,
        height=320,
        width=320,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state['canvas_key']}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with workspace_col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Processed Image")
    st.caption("This is what the neural network actually sees after preprocessing.")
    processed_img_container = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("How does this work?", expanded=False):
    st.markdown(
        """
        1. The canvas output is cropped, centered on a 28×28 grid (like MNIST), and normalized.
        2. The NumPy neural network runs a forward pass and computes class probabilities.
        3. Results are rendered instantly with a bar chart so you can inspect confidence for each digit.
        """
    )

# -----------------------------
# CANVAS PREPROCESSING
# -----------------------------
def _shift_to_center(img: Image.Image) -> Image.Image:
    """Shift the digit's center of mass to the middle of the 28×28 canvas."""
    arr = np.array(img, dtype=np.float32)
    mass = np.sum(arr)
    if mass == 0:
        return img

    rows = np.arange(arr.shape[0], dtype=np.float32)
    cols = np.arange(arr.shape[1], dtype=np.float32)
    row_center = float(np.dot(arr.sum(axis=1), rows) / mass)
    col_center = float(np.dot(arr.sum(axis=0), cols) / mass)
    target = (arr.shape[0] - 1) / 2.0
    shift_r = int(np.round(target - row_center))
    shift_c = int(np.round(target - col_center))

    if shift_r == 0 and shift_c == 0:
        return img

    shifted = np.roll(arr, shift_r, axis=0)
    if shift_r > 0:
        shifted[:shift_r, :] = 0
    elif shift_r < 0:
        shifted[shift_r:, :] = 0

    shifted = np.roll(shifted, shift_c, axis=1)
    if shift_c > 0:
        shifted[:, :shift_c] = 0
    elif shift_c < 0:
        shifted[:, shift_c:] = 0

    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return Image.fromarray(shifted, mode="L")


def preprocess_canvas(alpha: np.ndarray, blur: int) -> tuple[np.ndarray | None, Image.Image | None]:
    """Crop, resize, center, and normalize the alpha channel to mimic MNIST formatting."""
    if alpha is None or np.max(alpha) == 0:
        return None, None

    mask_indices = np.argwhere(alpha > 0)
    if mask_indices.size == 0:
        return None, None

    y_min, x_min = mask_indices.min(axis=0)
    y_max, x_max = mask_indices.max(axis=0)

    digit = alpha[y_min : y_max + 1, x_min : x_max + 1]
    digit_img = Image.fromarray(digit, mode="L")
    digit_img = ImageOps.autocontrast(digit_img)

    target_dim = 20
    max_dim = max(digit_img.size)
    if max_dim == 0:
        return None, None

    scale = target_dim / max_dim
    new_size = (
        max(1, int(round(digit_img.size[0] * scale))),
        max(1, int(round(digit_img.size[1] * scale))),
    )
    digit_img = digit_img.resize(new_size, Image.Resampling.LANCZOS)

    canvas_img = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas_img.paste(digit_img, offset)

    if blur > 0:
        canvas_img = canvas_img.filter(ImageFilter.GaussianBlur(blur))

    canvas_img = _shift_to_center(canvas_img)

    arr = np.array(canvas_img, dtype=np.float32).reshape(784, 1) / 255.0
    return arr, canvas_img


# -----------------------------
# PROCESS CANVAS INPUT
# -----------------------------
prediction_ready = False
arr = None
if canvas.image_data is not None:
    raw = canvas.image_data.astype("uint8")
    alpha = raw[:, :, -1]
    arr, processed_img = preprocess_canvas(alpha, blur_amount)

    if arr is not None and float(np.max(arr)) > 0.01:
        prediction_ready = True
        with workspace_col2:
            processed_img_container.image(
                processed_img.resize((200, 200), Image.Resampling.NEAREST),
                clamp=True,
            )

if prediction_ready and arr is not None:
    probs = nn.predict_proba(arr)
    pred = int(np.argmax(probs))
    flat_probs = probs.flatten()
    prob_df = pd.DataFrame(
        {
            "digit": np.arange(10),
            "confidence": flat_probs,
            "percent": flat_probs * 100,
        }
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Step 2 · Model Prediction")
    metric_col, chart_col = st.columns([1.1, 2])

    with metric_col:
        st.markdown('<div class="pill">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="confidence-display">
                <div class="confidence-digit">{pred}</div>
                <div class="confidence-info">
                    <span>Most probable digit</span>
                    <strong>{float(np.max(probs))*100:0.1f}% confidence</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_idx = np.argsort(flat_probs)[::-1][:3]
        cards = ['<div class="top-guess-grid">']
        for rank, idx in enumerate(top_idx, start=1):
            accent = " accent" if idx == pred else ""
            cards.append(
                f"""
                <div class="top-guess{accent}">
                    <div class="top-guess-rank">#{rank} Guess</div>
                    <div class="top-guess-digit">{int(idx)}</div>
                    <div class="top-guess-confidence">{flat_probs[idx]*100:0.1f}%</div>
                </div>
                """
            )
        cards.append("</div>")
        st.markdown("**Top guesses**", unsafe_allow_html=True)
        st.markdown("".join(cards), unsafe_allow_html=True)

    with chart_col:
        bars = (
            alt.Chart(prob_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusBottomLeft=6)
            .encode(
                y=alt.Y("digit:O", sort="-x", title="Digit"),
                x=alt.X(
                    "confidence:Q",
                    title="Confidence",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1], nice=False),
                ),
                color=alt.condition(
                    alt.datum.digit == pred,
                    alt.value("#2563eb"),
                    alt.value("#e0e7ff"),
                ),
                tooltip=[
                    alt.Tooltip("digit:O", title="Digit"),
                    alt.Tooltip("percent:Q", title="Confidence", format=".1f"),
                ],
            )
            .properties(height=330)
        )
        labels = (
            alt.Chart(prob_df)
            .mark_text(dx=6, fontSize=13, fontWeight="bold")
            .encode(
                y=alt.Y("digit:O", sort="-x"),
                x=alt.X("confidence:Q"),
                text=alt.Text("percent:Q", format=".1f"),
                color=alt.condition(
                    alt.datum.digit == pred,
                    alt.value("#0f172a"),
                    alt.value("#475569"),
                ),
            )
        )
        styled_chart = (
            (bars + labels)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#475569", titleColor="#1e293b", gridColor="#e2e8f0")
        )
        st.markdown('<div class="chart-title">Confidence per digit</div>', unsafe_allow_html=True)
        st.altair_chart(styled_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    processed_img_container.empty()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Step 2 · Model Prediction")
    st.info("Draw a digit or adjust the brush settings to start the prediction pipeline.")
    st.markdown("</div>", unsafe_allow_html=True)
