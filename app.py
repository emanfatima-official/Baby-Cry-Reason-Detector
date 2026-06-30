import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model
import joblib
import tempfile
import os
import queue
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# -----------------------------------------------------
# Page Configuration - MUST BE FIRST
# -----------------------------------------------------
st.set_page_config(
    page_title="Baby Cry Reason Detector",
    page_icon=":material/graphic_eq:",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
<style>
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
    .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        visibility: hidden !important;
        display: none !important;
    }
    .stMarkdown h1::before, .stMarkdown h2::before, .stMarkdown h3::before,
    .stMarkdown h4::before, .stMarkdown h5::before, .stMarkdown h6::before {
        content: none !important;
    }

    .bcrd-hero {
        background: linear-gradient(135deg, #2E3A59 0%, #4A5C8A 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.6rem;
        color: #F5F6FA;
    }
    .bcrd-hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .bcrd-hero p {
        margin: 0;
        font-size: 1.02rem;
        color: #D6DAEA;
    }

    .bcrd-card {
        background: #FFFFFF;
        border: 1px solid #E4E6EE;
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(20,20,40,0.04);
    }

    .bcrd-result-card {
        border-radius: 12px;
        padding: 1.5rem 1.6rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .bcrd-result-positive {
        background: #EAF7EE;
        border: 1px solid #B9E4C4;
    }
    .bcrd-result-warning {
        background: #FFF6E6;
        border: 1px solid #F3D9A0;
    }
    .bcrd-result-negative {
        background: #FBEAEA;
        border: 1px solid #F0BFBF;
    }
    .bcrd-result-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .bcrd-result-sub {
        font-size: 0.95rem;
        color: #4A4A4A;
    }

    .bcrd-suggestion-item {
        padding: 0.55rem 0.7rem;
        border-left: 3px solid #4A5C8A;
        background: #F7F8FC;
        margin-bottom: 0.45rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.95rem;
    }

    .bcrd-status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .bcrd-status-recording {
        background: #FBEAEA;
        color: #B33A3A;
    }
    .bcrd-status-stopped {
        background: #EAF7EE;
        color: #2E7D44;
    }

    .bcrd-footer {
        text-align: center;
        color: #8A8A8A;
        font-size: 0.85rem;
        margin-top: 2rem;
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
# Audio Validation (cry vs random noise / silence)
# -----------------------------------------------------
def is_baby_cry_audio(audio, sr=22050):
    if audio is None or len(audio) == 0:
        return False, "No audio captured."

    duration = len(audio) / sr
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Last Analysis Diagnostics")
        st.write(f"Duration: {duration:.2f}s")
        st.write(f"Max amplitude: {max_amp:.6f}")
        st.write(f"RMS energy: {rms:.6f}")
        st.write(f"Zero-crossing rate: {zcr:.4f}")
        st.write(f"Spectral centroid: {spec_centroid:.1f} Hz")
        st.write(f"Spectral flatness: {spec_flatness:.4f}")

    if rms < 0.0005:
        return False, "Audio is essentially silent."
    if max_amp < 0.001:
        return False, "Audio amplitude too low to analyze."
    if np.std(audio) < 0.0001:
        return False, "No meaningful variation detected in audio."

    # Crying has a fairly narrow, energetic, tonal band — very flat/noisy
    # spectra (fans, static, white noise, claps) get rejected here.
    if spec_flatness > 0.45:
        return False, "Sound looks like flat/white noise, not a baby cry."
    if spec_centroid < 150 or spec_centroid > 4500:
        return False, "Sound's frequency profile doesn't match a baby cry."

    return True, "Audio contains sound-like content - proceeding to analysis."

# -----------------------------------------------------
# Prediction Function
# -----------------------------------------------------
def predict_audio(audio, sr=22050):
    is_sound, sound_message = is_baby_cry_audio(audio, sr)

    if not is_sound:
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
        y=audio_processed,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred = model.predict(mfcc, verbose=0)
    reason = le.classes_[np.argmax(pred)]
    confidence = np.max(pred)

    with st.sidebar:
        st.markdown("### Prediction Probabilities")
        for i, class_name in enumerate(le.classes_):
            st.write(f"{class_name}: {pred[0][i] * 100:.1f}%")

    if confidence < 0.35:
        return "No Cry Detected", confidence, "Sound detected but it does not resemble a baby cry pattern."

    return reason, confidence, "Analysis completed."

# -----------------------------------------------------
# Suggestions
# -----------------------------------------------------
def get_suggestions(reason, confidence):
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
    st.markdown("### Analysis Results")

    if reason == "No Cry Detected":
        st.markdown(f"""
        <div class="bcrd-result-card bcrd-result-negative">
            <div class="bcrd-result-title">No Cry Detected</div>
            <div class="bcrd-result-sub">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Troubleshooting tips"):
            st.write("- Check that microphone permissions are granted in the browser.")
            st.write("- Make sure the baby is crying loudly enough during recording.")
            st.write("- Hold the microphone 10-20 cm from the baby for best results.")
            st.write("- Try uploading a WAV file instead of live recording.")
    else:
        if confidence < 0.6:
            st.markdown(f"""
            <div class="bcrd-result-card bcrd-result-warning">
                <div class="bcrd-result-title">Possible Reason: {reason}</div>
                <div class="bcrd-result-sub">Confidence: {confidence * 100:.1f}% (low confidence - try a clearer recording)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bcrd-result-card bcrd-result-positive">
                <div class="bcrd-result-title">Baby Cry Detected: {reason}</div>
                <div class="bcrd-result-sub">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Suggestions & Next Steps")
    for s in get_suggestions(reason, confidence):
        st.markdown(f'<div class="bcrd-suggestion-item">{s}</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown("""
<div class="bcrd-hero">
    <h1>Baby Cry Reason Detector</h1>
    <p>Record or upload a baby's cry to identify the likely reason — hungry, tired, or uncomfortable.</p>
</div>
""", unsafe_allow_html=True)

if model is None or le is None:
    st.error("Model files not loaded. Please ensure 'baby_cry_reason_model.h5' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

option = st.radio(
    "Choose input method:",
    ["Record via Microphone (Browser)", "Upload Audio File"],
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------
# Record Section - WebRTC Browser Mic
# -----------------------------------------------------
if option == "Record via Microphone (Browser)":
    st.markdown('<div class="bcrd-card">', unsafe_allow_html=True)
    st.subheader("Record Audio")
    st.write("1. Click **START** to allow microphone access.")
    st.write("2. Let the baby cry for a few seconds.")
    st.write("3. Click **STOP** when done.")
    st.write("4. Click **Analyze Recorded Audio** to get results.")
    st.markdown('</div>', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="baby-cry-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Reset the frame buffer whenever a fresh recording session starts.
    if ctx.state.playing and not st.session_state.get("bcrd_was_playing", False):
        st.session_state["bcrd_audio_frames"] = []
    st.session_state["bcrd_was_playing"] = ctx.state.playing

    if ctx.audio_receiver and ctx.state.playing:
        st.markdown('<span class="bcrd-status-pill bcrd-status-recording">Recording in progress...</span>', unsafe_allow_html=True)
        if "bcrd_audio_frames" not in st.session_state:
            st.session_state["bcrd_audio_frames"] = []

        # Keep pulling frames while the stream is active. This loop exits
        # naturally once the user clicks STOP (ctx.state.playing becomes False).
        while ctx.state.playing:
            try:
                audio_frames = ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                continue
            for frame in audio_frames:
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                st.session_state["bcrd_audio_frames"].append(arr.astype(np.float32))

        if st.session_state["bcrd_audio_frames"]:
            st.session_state["recorded_audio"] = np.concatenate(st.session_state["bcrd_audio_frames"])

    if "recorded_audio" in st.session_state and not ctx.state.playing:
        st.markdown('<span class="bcrd-status-pill bcrd-status-stopped">Recording captured - ready to analyze</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Recorded Audio", type="primary", use_container_width=True):
            audio_data = st.session_state["recorded_audio"]
            if len(audio_data) > 0:
                audio_resampled = librosa.resample(audio_data, orig_sr=48000, target_sr=sample_rate)
                with st.spinner("Analyzing audio..."):
                    reason, confidence, message = predict_audio(audio_resampled, sample_rate)
                show_results(reason, confidence, message)
            else:
                st.error("No audio was captured. Please try recording again.")

# -----------------------------------------------------
# Upload Section
# -----------------------------------------------------
elif option == "Upload Audio File":
    st.markdown('<div class="bcrd-card">', unsafe_allow_html=True)
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file containing baby crying", type=["wav"])
    st.markdown('</div>', unsafe_allow_html=True)

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
