import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model
import joblib
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# -----------------------------------------------------
# Page Configuration - MUST BE FIRST
# -----------------------------------------------------
st.set_page_config(
    page_title="Baby Cry Reason Detector",
    page_icon=":material/graphic_eq:",
    layout="centered"
)

# -----------------------------------------------------
# Custom CSS - simple, consistent, high-contrast
# -----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none !important;
    }

    /* ---------- Layout shell ---------- */
    .block-container {
        max-width: 760px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */
    .bcrd-header {
        text-align: center;
        padding-bottom: 1.6rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .bcrd-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 0.4rem 0;
    }
    .bcrd-header p {
        font-size: 0.98rem;
        color: #6B7280;
        margin: 0;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        font-weight: 600;
        color: #6B7280;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
    }

    /* ---------- Section card ---------- */
    .bcrd-section {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.4rem 1.5rem;
        margin: 1.2rem 0;
    }
    .bcrd-section h3 {
        font-size: 1.02rem;
        font-weight: 600;
        color: #111827;
        margin: 0 0 0.7rem 0;
    }
    .bcrd-step {
        font-size: 0.92rem;
        color: #4B5563;
        margin: 0.25rem 0;
    }

    /* ---------- Recorder row ---------- */
    .bcrd-recorder-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 1.6rem;
        background: #FFFFFF;
        border: 1px dashed #D1D5DB;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        width: 100%;
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.95rem;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
    }

    /* ---------- Result cards ---------- */
    .bcrd-result {
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin: 1.2rem 0 0.8rem 0;
        border-left: 5px solid;
    }
    .bcrd-result-positive {
        background: #F0FDF4;
        border-left-color: #16A34A;
    }
    .bcrd-result-warning {
        background: #FFFBEB;
        border-left-color: #D97706;
    }
    .bcrd-result-negative {
        background: #FEF2F2;
        border-left-color: #DC2626;
    }
    .bcrd-result-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .bcrd-result-positive .bcrd-result-eyebrow { color: #15803D; }
    .bcrd-result-warning .bcrd-result-eyebrow { color: #B45309; }
    .bcrd-result-negative .bcrd-result-eyebrow { color: #B91C1C; }

    .bcrd-result-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 0.25rem 0;
    }
    .bcrd-result-sub {
        font-size: 0.92rem;
        color: #4B5563;
        margin: 0;
    }

    /* ---------- Confidence bar ---------- */
    .bcrd-confidence-track {
        background: #E5E7EB;
        border-radius: 6px;
        height: 8px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    .bcrd-confidence-fill {
        height: 100%;
        border-radius: 6px;
    }

    /* ---------- Suggestions ---------- */
    .bcrd-suggestions-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
        margin: 1.4rem 0 0.6rem 0;
    }
    .bcrd-suggestion-item {
        display: flex;
        gap: 0.6rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid #F1F2F4;
        font-size: 0.92rem;
        color: #374151;
    }
    .bcrd-suggestion-item:last-child {
        border-bottom: none;
    }
    .bcrd-bullet {
        color: #2563EB;
        font-weight: 700;
    }

    /* ---------- Status pill ---------- */
    .bcrd-pill {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .bcrd-pill-info { background: #EFF6FF; color: #1D4ED8; }

    /* ---------- Footer ---------- */
    .bcrd-footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.82rem;
        margin-top: 2.4rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E5E7EB;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Model and Encoder
# -----------------------------------------------------
@st.cache_resource
def load_all():
    try:
        model = load_model("baby_cry_reason_model.h5", compile=False)
        le = joblib.load("label_encoder.pkl")
        return model, le, None
    except Exception as e:
        return None, None, str(e)

model, le, load_error = load_all()

with st.sidebar:
    st.markdown("### System Status")
    if model is not None and le is not None:
        st.success("Model and encoder loaded")
        st.write("**Loaded classes:**")
        for c in le.classes_:
            st.write(f"- {c}")
    else:
        st.error(f"Could not load model files: {load_error}")

sample_rate = 22050

# -----------------------------------------------------
# Cry validation - layered checks before trusting the model
# -----------------------------------------------------
def analyze_audio_characteristics(audio, sr=22050):
    """Returns a dict of acoustic features used to decide if this is
    plausibly a baby cry, before the classifier is even trusted."""
    duration = len(audio) / sr
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # Pitch / harmonicity: a baby cry is a strongly voiced, tonal sound
    # with a fundamental typically in the ~250-700 Hz range (higher than
    # adult speech). Random noise, claps, taps, or silence won't show
    # a stable pitch in this band.
    voiced_fraction = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, sr=sr,
            fmin=80, fmax=1000,
            frame_length=2048
        )
        if f0 is not None and len(f0) > 0:
            in_range = voiced_flag & (f0 >= 250) & (f0 <= 750)
            voiced_fraction = float(np.sum(in_range)) / float(len(f0))
    except Exception:
        voiced_fraction = 0.0

    return {
        "duration": duration,
        "max_amp": max_amp,
        "rms": rms,
        "zcr": zcr,
        "spec_centroid": spec_centroid,
        "spec_flatness": spec_flatness,
        "voiced_fraction": voiced_fraction,
    }


def is_baby_cry_audio(audio, sr=22050):
    if audio is None or len(audio) == 0:
        return False, "No audio captured.", None

    feats = analyze_audio_characteristics(audio, sr)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Diagnostics")
        st.write(f"Duration: {feats['duration']:.2f}s")
        st.write(f"RMS energy: {feats['rms']:.6f}")
        st.write(f"Spectral centroid: {feats['spec_centroid']:.1f} Hz")
        st.write(f"Spectral flatness: {feats['spec_flatness']:.4f}")
        st.write(f"Cry-band voiced fraction: {feats['voiced_fraction']:.2%}")

    if feats["rms"] < 0.0008:
        return False, "Audio is essentially silent.", feats
    if feats["max_amp"] < 0.002:
        return False, "Audio amplitude too low to analyze.", feats
    if np.std(audio) < 0.0001:
        return False, "No meaningful variation detected in audio.", feats
    if feats["spec_flatness"] > 0.4:
        return False, "Sound resembles flat/white noise, not a baby cry.", feats
    if feats["spec_centroid"] < 150 or feats["spec_centroid"] > 4500:
        return False, "Sound's frequency profile doesn't match a baby cry.", feats

    # Core gate: a real cry should show a sustained tonal pitch in the
    # infant cry band for a meaningful portion of the clip.
    if feats["voiced_fraction"] < 0.12:
        return False, "No sustained crying pitch pattern detected.", feats

    return True, "Audio matches expected baby cry characteristics.", feats

# -----------------------------------------------------
# Prediction Function
# -----------------------------------------------------
def predict_audio(audio, sr=22050):
    is_cry_like, sound_message, feats = is_baby_cry_audio(audio, sr)

    if not is_cry_like:
        return "No Cry Detected", 0.0, sound_message

    audio_processed = audio.astype(np.float32)

    if np.max(np.abs(audio_processed)) > 0:
        audio_processed = audio_processed / np.max(np.abs(audio_processed))

    audio_processed = audio_processed - np.mean(audio_processed)

    rms = np.sqrt(np.mean(audio_processed ** 2))
    if rms < 0.01:
        try:
            audio_processed = nr.reduce_noise(y=audio_processed, sr=sr, stationary=True, prop_decrease=0.3)
        except Exception:
            pass

    target_length = sr * 3
    if len(audio_processed) > target_length:
        audio_processed = audio_processed[:target_length]
    else:
        pad_length = target_length - len(audio_processed)
        audio_processed = np.pad(audio_processed, (0, pad_length), mode='constant')

    mfcc = librosa.feature.mfcc(
        y=audio_processed, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512
    )
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred = model.predict(mfcc, verbose=0)
    probs = pred[0]
    sorted_idx = np.argsort(probs)[::-1]
    top_idx, second_idx = sorted_idx[0], sorted_idx[1]
    confidence = probs[top_idx]
    margin = probs[top_idx] - probs[second_idx]
    reason = le.classes_[top_idx]

    with st.sidebar:
        st.markdown("### Prediction Probabilities")
        for i, class_name in enumerate(le.classes_):
            st.write(f"{class_name}: {probs[i] * 100:.1f}%")
        st.write(f"Margin (top vs runner-up): {margin * 100:.1f}%")

    # Reject low-confidence or ambiguous predictions instead of forcing
    # a label - this is what was causing random sounds to be mislabeled.
    if confidence < 0.45 or margin < 0.12:
        return "No Cry Detected", confidence, "Audio had cry-like qualities but the model could not confidently match a reason."

    return reason, confidence, "Analysis completed."

# -----------------------------------------------------
# Suggestions
# -----------------------------------------------------
def get_suggestions(reason):
    suggestions = {
        "Hungry": [
            "Check feeding schedule - baby might be due for a feed.",
            "Note the last feed time to see how long it has been.",
            "Look for hunger cues - rooting, lip-smacking, hand-to-mouth movements.",
            "Offer a feed - try breastfeeding or formula.",
            "Check hydration - make sure baby is getting enough fluids.",
        ],
        "Tired": [
            "Watch for sleep signals - eye-rubbing, yawning, fussiness.",
            "Check the sleep environment - room temperature, lighting, noise level.",
            "Track awake time - babies get overtired easily if kept up too long.",
            "Try a comfort routine - swaddling, gentle rocking, or white noise.",
            "Check for gas discomfort - may need burping or a gentle tummy rub.",
        ],
        "Uncomfortable": [
            "Check the diaper - it may be wet or soiled.",
            "Feel baby's neck - should be warm, not sweaty or cold.",
            "Check clothing - look for tight fits, tags, or rough fabric.",
            "Inspect skin - look for rashes, redness, or insect bites.",
            "Reposition baby - current position may be uncomfortable.",
        ],
        "No Cry Detected": [
            "No baby cry was identified in this recording.",
            "Try recording again with the microphone closer to the baby.",
            "Reduce background noise for a clearer recording.",
            "Make sure the baby is actively crying while recording.",
            "If using a file upload, confirm the WAV file actually contains crying audio.",
        ],
    }
    return suggestions.get(reason, [])

# -----------------------------------------------------
# Show Results Helper
# -----------------------------------------------------
def show_results(reason, confidence, message):
    if reason == "No Cry Detected":
        st.markdown(f"""
        <div class="bcrd-result bcrd-result-negative">
            <div class="bcrd-result-eyebrow">Result</div>
            <div class="bcrd-result-title">No Cry Detected</div>
            <p class="bcrd-result-sub">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Troubleshooting tips"):
            st.write("- Check that microphone permissions are granted in the browser.")
            st.write("- Make sure the baby is crying loudly enough during recording.")
            st.write("- Hold the microphone 10-20 cm from the baby for best results.")
            st.write("- Try uploading a WAV file instead of live recording.")
    else:
        if confidence < 0.6:
            css_class = "bcrd-result-warning"
            eyebrow = "Low Confidence"
            bar_color = "#D97706"
        else:
            css_class = "bcrd-result-positive"
            eyebrow = "Baby Cry Detected"
            bar_color = "#16A34A"

        st.markdown(f"""
        <div class="bcrd-result {css_class}">
            <div class="bcrd-result-eyebrow">{eyebrow}</div>
            <div class="bcrd-result-title">{reason}</div>
            <p class="bcrd-result-sub">Confidence: {confidence * 100:.1f}%</p>
            <div class="bcrd-confidence-track">
                <div class="bcrd-confidence-fill" style="width:{confidence * 100:.1f}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="bcrd-suggestions-title">Suggestions & Next Steps</div>', unsafe_allow_html=True)
    for s in get_suggestions(reason):
        st.markdown(f"""
        <div class="bcrd-suggestion-item">
            <span class="bcrd-bullet">&#8226;</span><span>{s}</span>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown("""
<div class="bcrd-header">
    <h1>Baby Cry Reason Detector</h1>
    <p>Record or upload audio to identify the likely reason a baby is crying.</p>
</div>
""", unsafe_allow_html=True)

if model is None or le is None:
    st.error("Model files not loaded. Please ensure 'baby_cry_reason_model.h5' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

tab_record, tab_upload = st.tabs(["Record Audio", "Upload File"])

# -----------------------------------------------------
# Record Tab
# -----------------------------------------------------
with tab_record:
    st.markdown("""
    <div class="bcrd-section">
        <h3>How to record</h3>
        <p class="bcrd-step">1. Click the microphone icon below to start recording.</p>
        <p class="bcrd-step">2. Let the baby cry for at least 3-5 seconds.</p>
        <p class="bcrd-step">3. Click the icon again to stop - analysis runs automatically.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="bcrd-recorder-row">', unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        text="",
        recording_color="#DC2626",
        neutral_color="#2563EB",
        icon_size="3x",
        sample_rate=sample_rate,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.session_state.get("bcrd_last_audio_bytes") != audio_bytes:
            st.session_state["bcrd_last_audio_bytes"] = audio_bytes

            with st.spinner("Analyzing audio..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    audio, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
                    reason, confidence, message = predict_audio(audio, sr)
                    st.session_state["bcrd_last_result"] = (reason, confidence, message)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.session_state["bcrd_last_result"] = None
                    st.error(f"Error processing recording: {e}")

        if st.session_state.get("bcrd_last_result"):
            reason, confidence, message = st.session_state["bcrd_last_result"]
            show_results(reason, confidence, message)
    else:
        st.markdown('<span class="bcrd-pill bcrd-pill-info">Waiting for recording</span>', unsafe_allow_html=True)

# -----------------------------------------------------
# Upload Tab
# -----------------------------------------------------
with tab_upload:
    st.markdown("""
    <div class="bcrd-section">
        <h3>Upload a WAV file</h3>
        <p class="bcrd-step">Choose a clear recording of a baby crying for best results.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select WAV file", type=["wav"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Analyze Uploaded Audio", type="primary", use_container_width=True):
            with st.spinner("Processing audio file..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    audio, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
                    reason, confidence, message = predict_audio(audio, sr)
                    show_results(reason, confidence, message)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error processing file: {e}")

st.markdown('<div class="bcrd-footer">Developed by Eman Fatima</div>', unsafe_allow_html=True)
