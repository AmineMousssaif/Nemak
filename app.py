from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib

from textwrap import dedent


def render_html(content: str) -> None:
    """Render custom HTML without Markdown interpreting it as code."""
    st.html(dedent(content).strip())

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nemak Vision AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "casting_model.keras"
METADATA_PATH = PROJECT_DIR / "outputs" / "model_metadata.json"

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_OK_THRESHOLD = 0.50
DEFAULT_REVIEW_MARGIN = 0.08


# =========================================================
# DESIGN SYSTEM
# =========================================================

st.html(

    """
<style>
/* -------------------------------------------------------
   GLOBAL RESET
------------------------------------------------------- */

:root {
    --bg: #070b14;
    --surface: rgba(15, 23, 38, 0.86);
    --surface-strong: #101827;
    --surface-soft: rgba(22, 34, 54, 0.72);
    --border: rgba(148, 163, 184, 0.16);
    --text: #f4f7fb;
    --muted: #8fa0b8;
    --teal: #36f1cd;
    --teal-dark: #0b8f82;
    --green: #35e59c;
    --red: #ff5263;
    --amber: #ffbd4a;
}

html,
body,
[class*="css"] {
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 5%,
            rgba(54, 241, 205, 0.10),
            transparent 24%
        ),
        radial-gradient(
            circle at 94% 18%,
            rgba(56, 128, 255, 0.09),
            transparent 28%
        ),
        linear-gradient(180deg, #070b14 0%, #0a101c 100%);
    color: var(--text);
}

#MainMenu,
footer,
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
}

/* -------------------------------------------------------
   TOP NAVIGATION
------------------------------------------------------- */

.app-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(10, 16, 28, 0.76);
    backdrop-filter: blur(18px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.brand-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    color: #06110f;
    background: linear-gradient(135deg, #36f1cd, #9cffec);
    font-size: 1.1rem;
    font-weight: 900;
    box-shadow: 0 0 24px rgba(54, 241, 205, 0.28);
}

.brand-name {
    color: var(--text);
    font-weight: 780;
    font-size: 0.98rem;
    letter-spacing: -0.01em;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 0.72rem;
    margin-top: 0.04rem;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.48rem;
    color: #b7fff1;
    background: rgba(54, 241, 205, 0.08);
    border: 1px solid rgba(54, 241, 205, 0.22);
    border-radius: 999px;
    padding: 0.44rem 0.75rem;
    font-size: 0.74rem;
    font-weight: 700;
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 5px rgba(54, 241, 205, 0.10);
    animation: pulse 1.7s infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }

    50% {
        opacity: 0.5;
        transform: scale(0.75);
    }
}

/* -------------------------------------------------------
   HERO
------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.4rem;
    border: 1px solid var(--border);
    border-radius: 26px;
    background:
        linear-gradient(
            135deg,
            rgba(16, 24, 39, 0.98),
            rgba(11, 20, 35, 0.90)
        );
    box-shadow:
        0 28px 70px rgba(0, 0, 0, 0.36),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.hero::after {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    right: -140px;
    top: -235px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(54, 241, 205, 0.18),
        transparent 66%
    );
}

.hero-eyebrow {
    color: var(--teal);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.hero-title {
    max-width: 840px;
    color: var(--text);
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: 1.03;
    font-weight: 850;
    letter-spacing: -0.045em;
    margin-bottom: 0.85rem;
}

.hero-title span {
    color: var(--teal);
}

.hero-copy {
    max-width: 780px;
    color: #a8b6ca;
    font-size: 1rem;
    line-height: 1.65;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1.25rem;
}

.hero-tag {
    color: #c8d4e6;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.42rem 0.68rem;
    font-size: 0.72rem;
}

/* -------------------------------------------------------
   PERFORMANCE CARDS
------------------------------------------------------- */

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 1rem 0;
}

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 112px;
    padding: 1rem 1.05rem;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--surface);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
}

.metric-card::after {
    content: "";
    position: absolute;
    right: -28px;
    top: -28px;
    width: 75px;
    height: 75px;
    border-radius: 50%;
    background: rgba(54, 241, 205, 0.06);
}

.metric-label {
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 760;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.metric-value {
    color: var(--text);
    font-size: 1.85rem;
    line-height: 1.1;
    font-weight: 820;
    letter-spacing: -0.035em;
    margin-top: 0.42rem;
}

.metric-caption {
    color: #71829a;
    font-size: 0.68rem;
    margin-top: 0.38rem;
}

/* -------------------------------------------------------
   MAIN WORKSPACE
------------------------------------------------------- */

.section-heading {
    margin: 1.7rem 0 0.75rem;
}

.section-kicker {
    color: var(--teal);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.section-title {
    color: var(--text);
    font-size: 1.45rem;
    font-weight: 790;
    letter-spacing: -0.025em;
    margin-top: 0.25rem;
}

.section-copy {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

.workspace-card {
    padding: 1.2rem;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: var(--surface);
    box-shadow: 0 22px 55px rgba(0, 0, 0, 0.27);
    min-height: 100%;
}

.workspace-label {
    display: flex;
    align-items: center;
    gap: 0.52rem;
    color: var(--teal);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.workspace-number {
    display: inline-grid;
    place-items: center;
    width: 25px;
    height: 25px;
    border-radius: 8px;
    color: #07120f;
    background: var(--teal);
    font-size: 0.66rem;
}

.workspace-title {
    color: var(--text);
    font-size: 1.28rem;
    font-weight: 790;
    margin-bottom: 0.28rem;
}

.workspace-copy {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
}

/* -------------------------------------------------------
   UPLOAD WIDGET
------------------------------------------------------- */

div[data-testid="stFileUploader"] {
    margin-top: 0.75rem;
}

div[data-testid="stFileUploaderDropzone"] {
    min-height: 150px;
    background:
        linear-gradient(
            135deg,
            rgba(54, 241, 205, 0.045),
            rgba(56, 128, 255, 0.025)
        ) !important;
    border: 1px dashed rgba(54, 241, 205, 0.42) !important;
    border-radius: 17px !important;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal) !important;
    background: rgba(54, 241, 205, 0.065) !important;
}

div[data-testid="stFileUploaderDropzone"] * {
    color: #b8c5d8 !important;
}

div[data-testid="stFileUploaderDropzone"] button {
    color: #06120f !important;
    background: linear-gradient(135deg, #36f1cd, #8affe5) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 22px rgba(54, 241, 205, 0.17);
}

div[data-testid="stFileUploaderFile"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.025);
}

div[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
}

/* -------------------------------------------------------
   EMPTY STATE
------------------------------------------------------- */

.empty-state {
    min-height: 330px;
    display: grid;
    place-items: center;
    text-align: center;
    padding: 2rem;
    border: 1px dashed rgba(148, 163, 184, 0.20);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.018);
}

.empty-icon {
    width: 72px;
    height: 72px;
    display: grid;
    place-items: center;
    margin: 0 auto 1rem;
    border-radius: 22px;
    color: var(--teal);
    background: rgba(54, 241, 205, 0.07);
    border: 1px solid rgba(54, 241, 205, 0.18);
    font-size: 1.65rem;
}

.empty-title {
    color: var(--text);
    font-size: 1.08rem;
    font-weight: 760;
}

.empty-copy {
    max-width: 320px;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.55;
    margin-top: 0.45rem;
}

/* -------------------------------------------------------
   DECISION CARDS
------------------------------------------------------- */

.decision-card {
    position: relative;
    overflow: hidden;
    padding: 1.35rem;
    border-radius: 19px;
    margin-top: 0.85rem;
}

.decision-card.ok {
    background:
        linear-gradient(
            135deg,
            rgba(53, 229, 156, 0.13),
            rgba(53, 229, 156, 0.035)
        );
    border: 1px solid rgba(53, 229, 156, 0.32);
}

.decision-card.defective {
    background:
        linear-gradient(
            135deg,
            rgba(255, 82, 99, 0.15),
            rgba(255, 82, 99, 0.035)
        );
    border: 1px solid rgba(255, 82, 99, 0.34);
}

.decision-card.review {
    background:
        linear-gradient(
            135deg,
            rgba(255, 189, 74, 0.14),
            rgba(255, 189, 74, 0.035)
        );
    border: 1px solid rgba(255, 189, 74, 0.34);
}

.decision-topline {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--muted);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.decision-status {
    color: var(--text);
    font-size: 2.35rem;
    line-height: 1.06;
    font-weight: 880;
    letter-spacing: -0.045em;
    margin: 0.55rem 0 0.45rem;
}

.decision-description {
    color: #a9b7ca;
    font-size: 0.84rem;
    line-height: 1.55;
}

.action-strip {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    margin-top: 0.8rem;
    padding: 0.8rem 0.9rem;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: rgba(255, 255, 255, 0.025);
}

.action-icon {
    flex: 0 0 auto;
    width: 31px;
    height: 31px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    color: #07120f;
    background: var(--teal);
    font-size: 0.9rem;
    font-weight: 900;
}

.action-title {
    color: var(--text);
    font-size: 0.76rem;
    font-weight: 740;
}

.action-copy {
    color: var(--muted);
    font-size: 0.72rem;
    line-height: 1.45;
    margin-top: 0.16rem;
}

/* -------------------------------------------------------
   PROBABILITY GAUGE
------------------------------------------------------- */

.gauge-layout {
    display: grid;
    grid-template-columns: 138px 1fr;
    align-items: center;
    gap: 1rem;
    padding-top: 1.1rem;
}

.gauge {
    --value: 0;
    --gauge-color: #36f1cd;

    width: 126px;
    height: 126px;
    position: relative;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background:
        conic-gradient(
            var(--gauge-color) calc(var(--value) * 1%),
            rgba(148, 163, 184, 0.12) 0
        );
}

.gauge::before {
    content: "";
    position: absolute;
    inset: 10px;
    border-radius: 50%;
    background: #0d1523;
    border: 1px solid rgba(148, 163, 184, 0.10);
}

.gauge-value {
    position: relative;
    z-index: 2;
    color: var(--text);
    font-size: 1.48rem;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -0.04em;
}

.gauge-label {
    position: relative;
    z-index: 2;
    color: var(--muted);
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.probability-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.68rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.10);
}

.probability-row:last-child {
    border-bottom: none;
}

.probability-name {
    color: var(--muted);
    font-size: 0.76rem;
}

.probability-value {
    color: var(--text);
    font-size: 0.82rem;
    font-weight: 760;
}

/* -------------------------------------------------------
   ADVANCED SETTINGS
------------------------------------------------------- */

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 38, 0.65) !important;
}

div[data-testid="stExpander"] summary {
    color: #ccd7e7 !important;
    font-size: 0.82rem !important;
}

/* -------------------------------------------------------
   WORKFLOW
------------------------------------------------------- */

.workflow-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 0.85rem;
}

.workflow-card {
    position: relative;
    padding: 1.05rem;
    border: 1px solid var(--border);
    border-radius: 17px;
    background: rgba(15, 23, 38, 0.72);
}

.workflow-number {
    color: var(--teal);
    font-size: 0.66rem;
    font-weight: 820;
    letter-spacing: 0.13em;
}

.workflow-title {
    color: var(--text);
    font-size: 0.98rem;
    font-weight: 760;
    margin: 0.42rem 0 0.25rem;
}

.workflow-copy {
    color: var(--muted);
    font-size: 0.73rem;
    line-height: 1.5;
}

/* -------------------------------------------------------
   FOOTER
------------------------------------------------------- */

.app-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: #667890;
    font-size: 0.68rem;
}

/* -------------------------------------------------------
   RESPONSIVE
------------------------------------------------------- */

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .workflow-grid {
        grid-template-columns: 1fr;
    }

    .gauge-layout {
        grid-template-columns: 1fr;
        justify-items: center;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .hero {
        padding: 1.5rem;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }

    .app-nav {
        align-items: flex-start;
    }

    .live-badge {
        display: none;
    }
}
</style>
    """
)


# =========================================================
# MODEL AND METADATA
# =========================================================

@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {
            "model": "MobileNetV2 transfer learning",
            "image_size": list(DEFAULT_IMAGE_SIZE),
        }

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# =========================================================
# INFERENCE FUNCTIONS
# =========================================================

def prepare_image(
    image: Image.Image,
    target_size: tuple[int, int],
) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(target_size)

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    return np.expand_dims(array, axis=0)


def predict_image(image: Image.Image) -> dict[str, float]:
    image_size = tuple(
        metadata.get(
            "image_size",
            list(DEFAULT_IMAGE_SIZE),
        )
    )

    model_input = prepare_image(
        image=image,
        target_size=image_size,
    )

    ok_probability = float(
        model.predict(
            model_input,
            verbose=0,
        )[0][0]
    )

    return {
        "ok_probability": ok_probability,
        "defect_probability": 1.0 - ok_probability,
    }

def find_feature_extractor(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Locate the nested convolutional base model.

    In this project, MobileNetV2 was added as a nested Keras model
    before global average pooling and the binary classification head.
    """

    nested_models = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    ]

    if not nested_models:
        raise ValueError(
            "No nested convolutional feature extractor was found."
        )

    # The largest nested model should be MobileNetV2.
    return max(
        nested_models,
        key=lambda nested_model: len(nested_model.layers),
    )

def make_gradcam_heatmap(
    model_input: np.ndarray,
    model: tf.keras.Model,
    target_class: str,
) -> np.ndarray:
    """
    Generate Grad-CAM for a model containing a nested MobileNetV2 base.

    The complete outer-model pipeline is rebuilt from model.inputs,
    ensuring that the convolutional feature maps and prediction output
    belong to the same connected graph.
    """

    feature_extractor = find_feature_extractor(model)

    # Rebuild the outer model forward path from its own input.
    outer_input = model.inputs[0]
    x = outer_input
    feature_maps = None

    for layer in model.layers[1:]:
        try:
            x = layer(x, training=False)
        except TypeError:
            # Some layers do not expose a training argument.
            x = layer(x)

        if layer is feature_extractor:
            feature_maps = x

    if feature_maps is None:
        raise RuntimeError(
            "The MobileNetV2 feature-map output could not be located "
            "in the outer model."
        )

    predictions = x

    grad_model = tf.keras.Model(
        inputs=outer_input,
        outputs=[
            feature_maps,
            predictions,
        ],
        name="gradcam_model",
    )

    input_tensor = tf.convert_to_tensor(
        model_input,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = grad_model(
            input_tensor,
            training=False,
        )

        ok_probability = tf.clip_by_value(
            model_predictions[:, 0],
            1e-7,
            1.0 - 1e-7,
        )

        # Use the pre-sigmoid equivalent score to avoid weak gradients
        # when the model outputs a probability extremely close to 0 or 1.
        ok_logit = tf.math.log(
            ok_probability / (1.0 - ok_probability)
        )

        if target_class == "ok":
            target_score = ok_logit
        else:
            target_score = -ok_logit

    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)

    if float(maximum.numpy()) > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()

def overlay_gradcam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    """
    Resize the Grad-CAM heatmap and overlay it on the original image.
    """

    base_image = original_image.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255),
        mode="L",
    ).resize(
        base_image.size,
        resample=Image.Resampling.BILINEAR,
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colormap = matplotlib.colormaps["turbo"]

    coloured_heatmap = colormap(
        heatmap_array,
    )[:, :, :3]

    coloured_heatmap = Image.fromarray(
        np.uint8(coloured_heatmap * 255),
        mode="RGB",
    )

    return Image.blend(
        base_image,
        coloured_heatmap,
        alpha=alpha,
    )


def generate_gradcam_explanation(
    image: Image.Image,
    status: str,
) -> Image.Image:
    """
    Generate the explanation for the current operational decision.
    """

    image_size = tuple(
        metadata.get(
            "image_size",
            list(DEFAULT_IMAGE_SIZE),
        )
    )

    model_input = prepare_image(
        image=image,
        target_size=image_size,
    )

    # For manual-review cases, explain the class with the larger probability.
    if status == "review":
        probability = predict_image(image)

        target_class = (
            "ok"
            if probability["ok_probability"]
            >= probability["defect_probability"]
            else "defective"
        )
    else:
        target_class = status

    heatmap = make_gradcam_heatmap(
        model_input=model_input,
        model=model,
        target_class=target_class,
    )

    return overlay_gradcam(
        original_image=image,
        heatmap=heatmap,
    )


def determine_status(
    ok_probability: float,
    threshold_ok: float,
    review_margin: float,
) -> str:
    if abs(ok_probability - threshold_ok) <= review_margin:
        return "review"

    if ok_probability >= threshold_ok:
        return "ok"

    return "defective"


def result_configuration(status: str) -> dict[str, str]:
    if status == "ok":
        return {
            "status": "Acceptable",
            "class": "ok",
            "symbol": "✓",
            "gauge_color": "#35e59c",
            "description": (
                "The component exceeds the selected quality threshold "
                "and may continue through production."
            ),
            "action_title": "Continue production",
            "action_copy": (
                "Allow the component to proceed, subject to standard "
                "quality-control procedures."
            ),
        }

    if status == "defective":
        return {
            "status": "Defect detected",
            "class": "defective",
            "symbol": "!",
            "gauge_color": "#ff5263",
            "description": (
                "The model identified visual patterns associated with "
                "a defective cast component."
            ),
            "action_title": "Isolate component",
            "action_copy": (
                "Remove the component from the line and route it to a "
                "quality inspector for confirmation."
            ),
        }

    return {
        "status": "Manual review",
        "class": "review",
        "symbol": "?",
        "gauge_color": "#ffbd4a",
        "description": (
            "The prediction is too close to the operating threshold for "
            "an automatic decision."
        ),
        "action_title": "Request human review",
        "action_copy": (
            "A quality inspector should assess the component before it "
            "continues through production."
        ),
    }


# =========================================================
# TOP NAVIGATION
# =========================================================

render_html(
    """
<div class="app-nav">
    <div class="brand-lockup">
        <div class="brand-icon">N</div>
        <div>
            <div class="brand-name">Nemak Vision AI</div>
            <div class="brand-subtitle">Industrial inspection prototype</div>
        </div>
    </div>

    <div class="live-badge">
        <span class="live-dot"></span>
        Model online
    </div>
</div>
    """
    )


# =========================================================
# HERO
# =========================================================

render_html(
    """
<div class="hero">
    <div class="hero-eyebrow">
        Intelligent manufacturing · Quality assurance
    </div>

    <div class="hero-title">
        See defects before they become
        <span>business risks.</span>
    </div>

    <div class="hero-copy">
        A CNN-powered visual inspection assistant for cast components.
        Upload an image to receive an immediate classification, confidence
        assessment and recommended production action.
    </div>

    <div class="hero-tags">
        <span class="hero-tag">MobileNetV2</span>
        <span class="hero-tag">Transfer learning</span>
        <span class="hero-tag">224 × 224 input</span>
        <span class="hero-tag">Human-in-the-loop</span>
    </div>
</div>
    """
)


# =========================================================
# PERFORMANCE OVERVIEW
# =========================================================

render_html(
    """
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Test accuracy</div>
        <div class="metric-value">93.85%</div>
        <div class="metric-caption">Across 715 test images</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">Defective-part recall</div>
        <div class="metric-value">95.81%</div>
        <div class="metric-caption">Priority business metric</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">Defects detected</div>
        <div class="metric-value">434</div>
        <div class="metric-caption">From 453 defective parts</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">Inference output</div>
        <div class="metric-value">&lt; 1 sec</div>
        <div class="metric-caption">Local prototype response</div>
    </div>
</div>
    """
)


# =========================================================
# ADVANCED CONTROLS
# =========================================================

with st.expander("Operating threshold and manual-review controls"):
    settings_left, settings_right = st.columns(2)

    with settings_left:
        threshold_ok = st.slider(
            "Minimum probability required to pass as acceptable",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_OK_THRESHOLD,
            step=0.01,
            help=(
                "A higher threshold makes automatic acceptance stricter."
            ),
        )

    with settings_right:
        review_margin = st.slider(
            "Manual-review safety margin",
            min_value=0.00,
            max_value=0.20,
            value=DEFAULT_REVIEW_MARGIN,
            step=0.01,
            help=(
                "Predictions near the threshold are routed to an inspector."
            ),
        )


# =========================================================
# INSPECTION WORKSPACE
# =========================================================

render_html(
    """
<div class="section-heading">
    <div class="section-kicker">Live prototype</div>
    <div class="section-title">Inspection workspace</div>
    <div class="section-copy">
        Upload a centred top-view casting image to run the CNN.
    </div>
</div>
    """
)

upload_column, result_column = st.columns(
    [1.03, 0.97],
    gap="large",
)

uploaded_image: Image.Image | None = None
gradcam_image: Image.Image | None = None
gradcam_error_message = ""

with upload_column:
    render_html(
        """
<div class="workspace-card">
    <div class="workspace-label">
        <span class="workspace-number">01</span>
        Image input
    </div>

    <div class="workspace-title">Upload casting component</div>

    <div class="workspace-copy">
        JPG, JPEG or PNG. Use a centred top-view image similar to
        the model's training conditions.
    </div>
</div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload casting image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)

            st.image(
                uploaded_image,
                caption=escape(uploaded_file.name),
                width="stretch",
            )

        except Exception:
            st.error("The uploaded file could not be read as an image.")


with result_column:
    render_html(
        """
<div class="workspace-card">
    <div class="workspace-label">
        <span class="workspace-number">02</span>
        AI decision
    </div>

    <div class="workspace-title">Inspection intelligence</div>

    <div class="workspace-copy">
        The result combines CNN probabilities, the operating threshold
        and the manual-review safety margin.
    </div>
</div>
        """
    )

    if uploaded_image is None:
        render_html(
            """
<div class="empty-state">
    <div>
        <div class="empty-icon">◎</div>
        <div class="empty-title">Waiting for component image</div>
        <div class="empty-copy">
            Upload an image on the left. The model will return a quality
            decision and recommended production action.
        </div>
    </div>
</div>
            """
        )

    else:
        with st.spinner("Running visual inspection..."):
            prediction = predict_image(uploaded_image)

        status = determine_status(
            ok_probability=prediction["ok_probability"],
            threshold_ok=threshold_ok,
            review_margin=review_margin,
        )

        config = result_configuration(status)

        try:
            gradcam_image = generate_gradcam_explanation(
                image=uploaded_image,
                status=status,
            )

        except Exception as gradcam_error:
            gradcam_image = None
            gradcam_error_message = str(gradcam_error)

       
        decision_confidence = (
            prediction["ok_probability"]
            if status == "ok"
            else prediction["defect_probability"]
        )

        confidence_percent = decision_confidence * 100
        defect_percent = prediction["defect_probability"] * 100
        ok_percent = prediction["ok_probability"] * 100

        render_html(
            f"""
<div class="decision-card {config['class']}">
    <div class="decision-topline">
        <span>{config['symbol']}</span>
        Inspection decision
    </div>

    <div class="decision-status">
        {escape(config['status'])}
    </div>

    <div class="decision-description">
        {escape(config['description'])}
    </div>

    <div class="action-strip">
        <div class="action-icon">→</div>
        <div>
            <div class="action-title">
                {escape(config['action_title'])}
            </div>

            <div class="action-copy">
                {escape(config['action_copy'])}
            </div>
        </div>
    </div>
</div>

<div class="gauge-layout">
    <div
        class="gauge"
        style="
            --value: {confidence_percent:.1f};
            --gauge-color: {config['gauge_color']};
        "
    >
        <div>
            <div class="gauge-value">
                {confidence_percent:.1f}%
            </div>

            <div class="gauge-label">
                confidence
            </div>
        </div>
    </div>

    <div>
        <div class="probability-row">
            <div class="probability-name">Defect probability</div>
            <div class="probability-value">{defect_percent:.1f}%</div>
        </div>

        <div class="probability-row">
            <div class="probability-name">Acceptable probability</div>
            <div class="probability-value">{ok_percent:.1f}%</div>
        </div>

        <div class="probability-row">
            <div class="probability-name">Pass threshold</div>
            <div class="probability-value">{threshold_ok:.2f}</div>
        </div>
    </div>
</div>
            """
        )


if uploaded_image is not None:
    st.write("")

    with st.expander(
        "Why did the model make this decision?",
        expanded=True,
    ):
        if gradcam_image is not None:
            explanation_left, explanation_right = st.columns(
                [1, 1],
                gap="medium",
            )

            with explanation_left:
                st.image(
                    uploaded_image,
                    caption="Original casting image",
                    width="stretch",
                )

            with explanation_right:
                st.image(
                    gradcam_image,
                    caption="Grad-CAM attention map",
                    width="stretch",
                )

            st.caption(
                "Warmer colours indicate regions that had a stronger "
                "influence on the model's classification."
            )

            st.info(
                "Grad-CAM shows where the CNN focused. It does not confirm "
                "that every highlighted region is a physical defect and "
                "should not replace inspector judgement."
            )

        else:
            st.warning(
                "The prediction was generated successfully, but the visual "
                f"explanation could not be created: {gradcam_error_message}"
            )


# =========================================================
# WORKFLOW
# =========================================================

render_html(
    """
<div class="section-heading">
    <div class="section-kicker">Operational integration</div>
    <div class="section-title">From camera to quality decision</div>
</div>

<div class="workflow-grid">
    <div class="workflow-card">
        <div class="workflow-number">STEP 01</div>
        <div class="workflow-title">Capture</div>
        <div class="workflow-copy">
            A standardised top-view image is captured at the inspection
            station.
        </div>
    </div>

    <div class="workflow-card">
        <div class="workflow-number">STEP 02</div>
        <div class="workflow-title">Analyse</div>
        <div class="workflow-copy">
            MobileNetV2 extracts visual features and estimates defect
            probability.
        </div>
    </div>

    <div class="workflow-card">
        <div class="workflow-number">STEP 03</div>
        <div class="workflow-title">Act</div>
        <div class="workflow-copy">
            Components continue or are isolated for inspection based on
            the operating policy.
        </div>
    </div>
</div>
    """
)


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
<div class="app-footer">
    <span>Nemak Vision AI · Academic prototype</span>
    <span>Neural Networks & Deep Learning Applied to Business</span>
</div>
    """
)