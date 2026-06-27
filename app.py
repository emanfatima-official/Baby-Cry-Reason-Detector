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
    page_icon="🎧",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
<style>
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a,
    .stMarkdown h5 a,
    .stMarkdown h6 a {
        visibility: hidden !important;
        display: none !important;
    }
    .stMarkdown h1:hover,
    .stMarkdown h2:hover,
    .stMarkdown h3:hover,
    .stMarkdown h4:hover,
    .stMarkdown h5:hover,
    .stMarkdown h6:hover {
        text-decoration: none !important;
    }
    .stMarkdown h1::before,
    .stMarkdown h2::before,
    .stMarkdown h3::before,
    .stMarkdown h4::before,
    .stMarkdown h5::before,
    .stMarkdown h6::before {
        content: none !important;
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
        st.sidebar.success("Model & Encoder loaded successfully!")
        return model, le
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, le = load_all()

if le is not None:
    st.sidebar.write("**Loaded Classes:**", list(le.classes_))

sample_rate = 22050

# -----------------------------------------------------
# Audio Validation
# -----------------------------------------------------
def is_baby_cry_audio(audio, sr=22050):
    if audio is None or len(audio) == 0:
        return False, "Empty audio"

    duration = len(audio) / sr
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))

    st.sidebar.write("**Audio Statistics:**")
    st.sidebar.write(f"Duration: {duration:.2f}s")
    st.sidebar.write(f"Max Amplitude: {max_amp:.6f}")
    st.sidebar.write(f"RMS: {rms:.6f}")

    if rms < 0.0005:
        return False, "Audio too quiet (silence detected)"
    if max_amp < 0.001:
        return False, "Very low audio amplitude"
    if np.std(audio) < 0.0001:
        return False, "No audio variation (flat line)"

    return True, "Audio contains sound - proceeding to analysis"

# -----------------------------------------------------
# Prediction Function
# -----------------------------------------------------
def predict_audio(audio, sr=22050):
    is_sound, sound_message = is_baby_cry_audio(audio, sr)

    if not is_sound:
        return "No Sound Detected", 0.0, sound_message

    audio_processed = audio.astype(np.float32)

    if np.max(np.abs(audio_processed)) > 0:
        audio_processed = audio_processed / np.max(np.abs(audio_processed))

    audio_processed = audio_processed - np.mean(audio_processed)

    rms = np.sqrt(np.mean(audio_processed**2))
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

    st.sidebar.write("**Prediction Probabilities:**")
    for i, class_name in enumerate(le.classes_):
        st.sidebar.write(f"  {class_name}: {pred[0][i]*100:.1f}%")

    if confidence < 0.3:
        return "Unclear Sound", confidence, "Low confidence - may not be baby crying"

    return reason, confidence, "Analysis completed"

# -----------------------------------------------------
# Suggestions
# -----------------------------------------------------
def get_suggestions(reason, confidence):
    suggestions = {
        "Hungry": [
            "Check feeding schedule - Baby might be hungry",
            "Last feed time - Note when baby was last fed",
            "Hunger cues - Look for rooting, lip-smacking, hand-to-mouth movements",
            "Offer feeding - Try breastfeeding or formula",
            "Check hydration - Ensure baby is getting enough fluids"
        ],
        "Tired": [
            "Sleep signals - Look for eye-rubbing, yawning, fussiness",
            "Sleep environment - Check room temperature, lighting, noise",
            "Awake time - Babies get overtired easily - check awake duration",
            "Comfort routine - Try swaddling, rocking, or white noise",
            "Gas discomfort - May need burping or gentle tummy massage"
        ],
        "Uncomfortable": [
            "Diaper check - Check for wet or soiled diaper",
            "Temperature - Feel baby's neck - should be warm, not sweaty or cold",
            "Clothing - Check for tight clothing, tags, or rough fabrics",
            "Skin irritation - Look for rashes, redness, or insect bites",
            "Position - Baby might be uncomfortable in current position"
        ],
        "Unclear Sound": [
            "The audio was unclear or may not contain baby crying",
            "Try recording again with clearer audio",
            "Ensure baby is actually crying during recording",
            "Hold microphone closer to the baby",
            "Record in a quieter environment"
        ]
    }
    return suggestions.get(reason, [])

# -----------------------------------------------------
# Show Results Helper
# -----------------------------------------------------
def show_results(reason, confidence, message):
    st.markdown("---")
    st.subheader("Analysis Results")

    if reason == "No Sound Detected":
        st.error("NO AUDIO DETECTED")
        st.write(f"**Reason:** {message}")
        st.info("""
        **Troubleshooting tips:**
        - Check if microphone permissions are granted in browser
        - Ensure baby is crying loudly enough
        - Move closer to the baby (10-20 cm ideal)
        - Try uploading a WAV file instead
        """)
    elif reason == "Unclear Sound":
        st.warning("UNCLEAR AUDIO")
        st.write(f"Confidence: {confidence*100:.1f}%")
        st.write("The audio was detected but may not contain clear baby crying sounds.")
    else:
        if confidence < 0.6:
            st.warning(f"LOW CONFIDENCE: {reason}")
            st.write(f"Confidence: {confidence*100:.1f}%")
            st.info("Baby cry detected but prediction confidence is low. Try again with clearer audio.")
        else:
            st.success("BABY CRY DETECTED")
            st.success(f"**Predicted Reason: {reason}**")
            st.write(f"**Confidence: {confidence*100:.1f}%**")

    st.markdown("---")
    st.subheader("Suggestions & Next Steps")
    for s in get_suggestions(reason, confidence):
        st.write(f"• {s}")

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.title("Baby Cry Reason Detector")
st.markdown("Detect why your baby is crying from audio recordings.")
st.markdown("---")

if model is None or le is None:
    st.error("Model files not loaded. Please ensure 'baby_cry_reason_model.h5' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

option = st.radio(
    "Choose Input Method:",
    ["Record via Microphone (Browser)", "Upload Audio File"],
    horizontal=True
)

# -----------------------------------------------------
# Record Section - WebRTC Browser Mic
# -----------------------------------------------------
if option == "Record via Microphone (Browser)":
    st.subheader("Record Audio")

    st.info("""
    **Instructions:**
    1. Click **START** below to allow microphone access
    2. Let the baby cry for a few seconds
    3. Click **STOP** when done
    4. Click **Analyze Recorded Audio** to get results
    """)

    audio_queue = queue.Queue()

    def audio_callback(frame: av.AudioFrame):
        audio_array = frame.to_ndarray()
        # Convert to mono float32
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)
        audio_array = audio_array.astype(np.float32)
        audio_queue.put(audio_array)
        return frame

    ctx = webrtc_streamer(
        key="baby-cry-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx.audio_receiver:
        audio_chunks = []
        st.write("🔴 Recording... click STOP when done.")
        try:
            while True:
                try:
                    audio_frames = ctx.audio_receiver.get_frames(timeout=1)
                    for frame in audio_frames:
                        arr = frame.to_ndarray()
                        if arr.ndim > 1:
                            arr = arr.mean(axis=0)
                        audio_chunks.append(arr.astype(np.float32))
                except queue.Empty:
                    break
        except Exception:
            pass

        if audio_chunks:
            st.session_state["recorded_audio"] = np.concatenate(audio_chunks)
            st.success("Recording saved! Click Analyze below.")

    if "recorded_audio" in st.session_state:
        if st.button("Analyze Recorded Audio", type="primary", use_container_width=True):
            audio_data = st.session_state["recorded_audio"]
            # Resample to 22050 if needed (WebRTC usually gives 48000)
            if len(audio_data) > 0:
                audio_resampled = librosa.resample(audio_data, orig_sr=48000, target_sr=sample_rate)
                with st.spinner("Analyzing audio..."):
                    reason, confidence, message = predict_audio(audio_resampled, sample_rate)
                show_results(reason, confidence, message)

# -----------------------------------------------------
# Upload Section
# -----------------------------------------------------
elif option == "Upload Audio File":
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose WAV file containing baby crying", type=["wav"])

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

st.markdown("---")
st.caption("Developed by Eman Fatima")
